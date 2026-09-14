"""diart streaming diarization on a Mac input device, published as JSON over WebSocket.

    uv run python -m live_diarizer --device "MacBook Pro Microphone" --port 8765

Messages (one JSON object per frame):

    {"type": "hello", "sample_rate": 16000, "latency": 1.0, "wall_offset": 1789400000.0}
    {"type": "turn", "speaker": "speaker0", "start": 12.3, "end": 14.1,
     "wall_start": ..., "wall_end": ..., "final": false}
    {"type": "step", "time": 14.5, "active": ["speaker0"]}

``start``/``end`` are seconds since the audio stream started; ``wall_*`` are the same instants
in ``time.time()`` seconds so the main process can align them with its own utterances.
Speaker labels are diart's anonymous ``speakerN``; naming them is the main process's job
(voiceprints). The gated pyannote models download with the cached Hugging Face login
(``hf auth login``) or ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Any

from dotenv import load_dotenv

from live_diarizer.turns import TurnTracker, annotation_segments

log = logging.getLogger("live_diarizer")


# --------------------------------------------------------------------------- devices
def resolve_device(spec: str | None) -> int | None:
    """Empty/None = default input; digits = index; otherwise a case-insensitive name substring."""
    import sounddevice as sd

    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if int(d["max_input_channels"]) > 0 and spec.lower() in d["name"].lower():
            return i
    names = ", ".join(
        f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0
    )
    raise SystemExit(f"no input device matching {spec!r}; available: {names}")


# --------------------------------------------------------------------------- broadcast
class Broadcaster:
    """WebSocket server on its own asyncio loop; ``publish`` is safe from any thread."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.clients: set[Any] = set()
        self.hello: dict | None = None
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="ws", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        import websockets

        async with websockets.serve(self._handler, self.host, self.port):
            log.info("publishing on ws://%s:%d", self.host, self.port)
            await asyncio.Future()

    async def _handler(self, websocket: Any) -> None:
        self.clients.add(websocket)
        try:
            if self.hello is not None:
                await websocket.send(json.dumps(self.hello))
            async for _ in websocket:  # clients never send anything; keep the socket open
                pass
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)

    def publish(self, message: dict) -> None:
        if message.get("type") == "hello":
            self.hello = message
        self.loop.call_soon_threadsafe(asyncio.ensure_future, self._send_all(json.dumps(message)))

    async def _send_all(self, payload: str) -> None:
        for client in list(self.clients):
            try:
                await client.send(payload)
            except Exception:
                self.clients.discard(client)


# --------------------------------------------------------------------------- diarizer
def _allow_pickled_checkpoints() -> None:
    """pyannote 3.x checkpoints are full pickles; torch >= 2.6 refuses them by default.

    They come from pyannote's own Hugging Face repositories (terms accepted with the login
    used here), so loading them with ``weights_only=False`` is the documented workaround.
    """
    import functools

    import torch

    original = torch.load
    if getattr(original, "_live_diarizer_patched", False):
        return

    @functools.wraps(original)
    def load(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is None:  # Lightning passes weights_only=None explicitly
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    load._live_diarizer_patched = True  # type: ignore[attr-defined]
    torch.load = load


def build_pipeline(args: argparse.Namespace):
    _allow_pickled_checkpoints()
    import torch
    from diart import SpeakerDiarization, SpeakerDiarizationConfig
    from diart.models import EmbeddingModel, SegmentationModel
    from huggingface_hub import get_token

    # Pass the token explicitly: with huggingface_hub 0.x, ``use_auth_token=True`` does not
    # authenticate the first download of a gated repository.
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("no Hugging Face token: run `hf auth login` or set HF_TOKEN in the repository .env")
    device = torch.device("mps") if args.mps and torch.backends.mps.is_available() else torch.device("cpu")
    config = SpeakerDiarizationConfig(
        segmentation=SegmentationModel.from_pretrained(args.segmentation, use_hf_token=token),
        embedding=EmbeddingModel.from_pretrained(args.embedding, use_hf_token=token),
        step=args.step,
        latency=args.latency,
        tau_active=args.tau_active,
        rho_update=args.rho_update,
        delta_new=args.delta_new,
        max_speakers=args.max_speakers,
        device=device,
    )
    log.info("pipeline on %s: step %.2fs latency %.2fs", device, args.step, args.latency)
    return SpeakerDiarization(config)


def run(args: argparse.Namespace) -> int:
    from diart.inference import StreamingInference
    from diart.sources import MicrophoneAudioSource

    broadcaster = Broadcaster(args.host, args.port)
    broadcaster.start()
    pipeline = build_pipeline(args)
    source = MicrophoneAudioSource(block_duration=args.step, device=resolve_device(args.device))
    log.info("listening on %s at %d Hz", source.uri, source.sample_rate)
    tracker = TurnTracker(gap=args.gap)
    wall_offset = time.time()
    broadcaster.publish(
        {
            "type": "hello",
            "sample_rate": source.sample_rate,
            "latency": args.latency,
            "step": args.step,
            "wall_offset": round(wall_offset, 3),
            "device": source.uri,
        }
    )
    state = {"steps": 0}

    def on_step(result: tuple) -> None:
        annotation, waveform = result
        step_end = float(waveform.extent.end)
        segments = annotation_segments(annotation)
        turns = tracker.update(segments, step_end)
        for turn in turns:
            broadcaster.publish(turn.to_message(wall_offset))
        active = sorted({t.speaker for t in turns if not t.final})
        broadcaster.publish({"type": "step", "time": round(step_end, 3), "active": active})
        state["steps"] += 1
        if args.verbose and (segments or state["steps"] % 20 == 0):
            log.info(
                "t=%.1f active=%s segments=%s",
                step_end,
                active,
                [(s, round(a, 1), round(b, 1)) for s, a, b in segments],
            )

    inference = StreamingInference(pipeline, source, do_profile=False, do_plot=False, show_progress=False)
    inference.attach_hooks(on_step)
    try:
        inference()
    except KeyboardInterrupt:
        pass
    finally:
        for turn in tracker.flush():
            broadcaster.publish(turn.to_message(wall_offset))
        source.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device", default=os.getenv("AUDIO_INPUT_DEVICE", ""), help="input device substring/index"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("DIARIZER_PORT", "8765")))
    parser.add_argument("--step", type=float, default=0.5, help="seconds of audio per inference step")
    parser.add_argument(
        "--latency", type=float, default=1.0, help="seconds of look-back before a label is emitted"
    )
    parser.add_argument("--gap", type=float, default=0.3, help="silence that ends a speaker turn")
    parser.add_argument("--tau-active", type=float, default=0.555)
    parser.add_argument("--rho-update", type=float, default=0.422)
    parser.add_argument("--delta-new", type=float, default=1.517)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument("--segmentation", default="pyannote/segmentation-3.0")
    parser.add_argument("--embedding", default="pyannote/embedding")
    parser.add_argument("--mps", action="store_true", help="run the models on the Apple GPU (default: CPU)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s", stream=sys.stdout
    )
    return run(args)
