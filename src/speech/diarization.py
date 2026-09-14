"""Client for the live diarizer sidecar (``tools/live-diarizer``): who spoke when.

    client = DiarizerClient()          # DIARIZER_URL; connects in the background, retries
    client.start()
    ...
    label = client.speaker_between(utterance_wall_start, utterance_wall_end)   # "speaker0" or ""

The sidecar publishes anonymous ``speakerN`` turns with ``wall_start`` / ``wall_end`` in
``time.time()`` seconds; :meth:`speaker_between` returns the label with the most overlap with
an interval. The client is optional: when the sidecar is not running the loop works without
speaker labels and logs that once.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass

from src.config import DIARIZER_URL
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class SpeakerTurn:
    speaker: str
    wall_start: float
    wall_end: float
    final: bool


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def dominant_speaker(turns: list[SpeakerTurn], start: float, end: float) -> tuple[str, float]:
    """Label with the most seconds of overlap with ``[start, end]`` and that overlap; ``("", 0)`` if none."""
    totals: dict[str, float] = {}
    for turn in turns:
        seconds = overlap(turn.wall_start, turn.wall_end, start, end)
        if seconds > 0:
            totals[turn.speaker] = totals.get(turn.speaker, 0.0) + seconds
    if not totals:
        return "", 0.0
    best = max(totals.items(), key=lambda kv: kv[1])
    return best


class TurnLog:
    """Rolling list of turns keyed by (speaker, start): updates replace, finals stick."""

    def __init__(self, keep_s: float = 300.0):
        self.keep_s = keep_s
        self._turns: dict[tuple[str, float], SpeakerTurn] = {}
        self.active: list[str] = []
        self.lock = threading.Lock()

    def apply(self, message: dict) -> None:
        kind = message.get("type")
        with self.lock:
            if kind == "turn":
                turn = SpeakerTurn(
                    speaker=str(message["speaker"]),
                    wall_start=float(message["wall_start"]),
                    wall_end=float(message["wall_end"]),
                    final=bool(message.get("final", False)),
                )
                self._turns[(turn.speaker, round(turn.wall_start, 2))] = turn
                self._prune(turn.wall_end)
            elif kind == "step":
                self.active = list(message.get("active", []))

    def _prune(self, now: float) -> None:
        for key in [k for k, t in self._turns.items() if now - t.wall_end > self.keep_s]:
            del self._turns[key]

    def turns(self) -> list[SpeakerTurn]:
        with self.lock:
            return sorted(self._turns.values(), key=lambda t: t.wall_start)

    def speaker_between(self, start: float, end: float, *, wait_s: float = 0.0) -> tuple[str, float]:
        """Dominant label over ``[start, end]``; waits up to ``wait_s`` for the sidecar to catch up.

        The sidecar labels audio about 1.5 s after it was heard (its latency plus a step), so
        a lookup right after an utterance ends may find nothing yet.
        """
        deadline = time.monotonic() + wait_s
        while True:
            label, seconds = dominant_speaker(self.turns(), start, end)
            if label or time.monotonic() >= deadline:
                return label, seconds
            time.sleep(0.1)


class DiarizerClient(TurnLog):
    """Background WebSocket reader with reconnection; the sidecar is optional."""

    def __init__(self, url: str = DIARIZER_URL, *, keep_s: float = 300.0, retry_s: float = 5.0):
        super().__init__(keep_s=keep_s)
        self.url = url
        self.retry_s = retry_s
        self.connected = False
        self.hello: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False

    def start(self) -> DiarizerClient:
        self._thread = threading.Thread(target=self._run, name="diarizer-client", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        import websockets

        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, open_timeout=3) as ws:
                    self.connected = True
                    log.info("diarizer connected at %s", self.url)
                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except TimeoutError:
                            continue
                        message = json.loads(raw)
                        if message.get("type") == "hello":
                            self.hello = message
                        self.apply(message)
            except Exception as exc:
                if not self._warned and not self._stop.is_set():
                    log.info(
                        "diarizer not available at %s (%s); continuing without speaker labels", self.url, exc
                    )
                    self._warned = True
            finally:
                self.connected = False
            await asyncio.sleep(self.retry_s)

    def wait_connected(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connected:
                return True
            time.sleep(0.1)
        return False


__all__ = ["DiarizerClient", "SpeakerTurn", "TurnLog", "dominant_speaker", "overlap"]
