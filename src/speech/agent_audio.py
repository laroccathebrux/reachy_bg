"""Audio plumbing between the ElevenLabs conversation and the table: Mac microphone in, robot USB speaker out.

    audio = RobotAudioInterface()                  # AUDIO_INPUT_DEVICE -> agent; agent -> AUDIO_OUTPUT_DEVICE
    audio.taps.append(lambda frame: ...)           # every 16 kHz int16 input frame, for local processing
    Conversation(..., audio_interface=audio)

The robot's speaker is a USB audio device on the Mac ("Reachy Mini Audio", 16 kHz); the SDK's
own client-side playback is silent on macOS, so agent audio is written straight to that device
from a playback thread. ``interrupt`` drops everything queued (barge-in) and ``level_db`` is the
loudness of what is being played right now, for head motion.

Two gates sit between the microphone and the agent:

* **Echo gate.** The Mac microphone has no echo cancellation and hears the robot at the same
  level as a player, so while the speaker is busy (plus a short tail) microphone audio is held
  back in a small ring buffer instead of going to the agent; the local taps still receive it.
  When the local echo-aware check decides a player is really talking over the robot,
  :meth:`release_gate` sends the held audio to the agent (so the phrase is not clipped) and
  stops playback. Without that check the agent would hear itself and answer itself in a loop.

* **Addressee gate** (``hold_utterances``). Everything that passes the echo gate is held in a
  second buffer, stamped with its arrival time, until the local ear decides whether the
  utterance was meant for the robot. :meth:`release_utterance` forwards the frames of one
  utterance in a burst (plus a short silent tail so the agent's turn detector closes the turn);
  :meth:`discard_utterance` drops them, and the agent never hears the sentence, so no cloud turn
  and no tokens are spent. :meth:`pass_through` opens the gate for the rest of the current
  utterance when the decision is already known (the robot's name was heard, or a barge-in).
  The agent sees audio chunks without timestamps, so a held-then-burst utterance is, to it,
  just a late one.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from src.config import AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE
from src.logger import get_logger
from src.speech.microphone import MicrophoneError, list_input_devices, resolve_input_device

log = get_logger(__name__)

SAMPLE_RATE = 16_000
INPUT_BLOCK = 4000  # 250 ms, what the agent expects
OUTPUT_BLOCK = 1600  # 100 ms
FRAME_S = INPUT_BLOCK / SAMPLE_RATE
HOLD_MAX_FRAMES = 120  # 30 s of held utterance audio (VAD_MAX_UTTERANCE_S is 20)
TURN_TAIL_S = 1.0  # silence appended to a released utterance so the agent closes the turn


def resolve_output_device(spec: str) -> int:
    """Index of the output device whose name contains ``spec`` (case-insensitive)."""
    import sounddevice as sd

    spec = (spec or "").strip().lower()
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if int(d["max_output_channels"]) > 0 and (not spec or spec in d["name"].lower()):
            return i
    names = ", ".join(f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_output_channels"] > 0)
    raise MicrophoneError(f"no output device matching {spec!r}; available: {names}")


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -100.0
    x = samples.astype(np.float64) / 32768.0
    rms = math.sqrt(float(np.mean(x * x)))
    return 20.0 * math.log10(rms) if rms > 0 else -100.0


class RobotAudioInterface:
    """``elevenlabs.conversational_ai.conversation.AudioInterface`` for a Mac mic and the robot speaker."""

    def __init__(
        self,
        input_device: str = AUDIO_INPUT_DEVICE,
        output_device: str = AUDIO_OUTPUT_DEVICE,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.input_spec = input_device
        self.output_spec = output_device
        self.clock = clock
        self.taps: list[Callable[[np.ndarray], None]] = []
        self.level_db = -100.0
        self.speaking = False
        self.muted = False
        self.gate_while_speaking = True
        self.gate_tail_s = 0.3
        self._spoke_until = 0.0  # monotonic time until which the echo gate stays closed
        self.last_played_at = 0.0  # monotonic time of the last block written to the speaker
        self._held: deque[bytes] = deque(maxlen=8)  # 8 x 250 ms of echo-gated microphone audio
        self.gated_frames = 0
        # Addressee gate: microphone audio waits here for the local decision.
        self.hold_utterances = False
        self.turn_tail_s = TURN_TAIL_S
        self._utterance: deque[tuple[float, bytes]] = deque(maxlen=HOLD_MAX_FRAMES)
        self._passing = False  # forward frames until the current utterance ends (+ tail)
        self._pass_frames_left = -1  # tail frames still to forward once the utterance ended
        self._lock = threading.Lock()
        self.held_frames = 0
        self.released_frames = 0
        self.discarded_frames = 0
        self._in: Any | None = None
        self._out: Any | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._player: threading.Thread | None = None
        self._stop = threading.Event()
        self._callback: Callable[[bytes], None] | None = None
        self.input_name = ""
        self.output_name = ""

    # ------------------------------------------------------------------ AudioInterface
    def start(self, input_callback: Callable[[bytes], None]) -> None:
        import sounddevice as sd

        self._callback = input_callback
        # input_spec None is "speak, do not listen": the rehearsal in scripts/speak_turn.py and
        # anything else that only needs the robot's voice opens no microphone at all.
        if self.input_spec is None:
            self.input_name = ""
        else:
            device = resolve_input_device(self.input_spec, list_input_devices(), sd.default.device[0])
            self.input_name = device.name
        out_index = resolve_output_device(self.output_spec)
        self.output_name = str(sd.query_devices(out_index)["name"])
        self._in = (
            None
            if self.input_spec is None
            else sd.InputStream(
                device=device.index,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype="int16",
                blocksize=INPUT_BLOCK,
                callback=self._on_input,
            )
        )
        self._out = sd.OutputStream(
            device=out_index, channels=1, samplerate=SAMPLE_RATE, dtype="int16", blocksize=OUTPUT_BLOCK
        )
        self._stop.clear()
        self._player = threading.Thread(target=self._play_loop, name="agent-playback", daemon=True)
        self._out.start()
        self._player.start()
        if self._in is not None:
            self._in.start()
        log.info("agent audio: %s -> agent -> %s", self.input_name or "(no microphone)", self.output_name)

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        for stream in (self._in, self._out):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as exc:
                    log.warning("closing an audio stream failed: %s", exc)
        self._in = self._out = None
        if self._player is not None:
            self._player.join(timeout=2.0)

    def output(self, audio: bytes) -> None:
        self._queue.put(audio)

    def interrupt(self) -> None:
        """Barge-in: drop everything queued and go quiet at the next block."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.level_db = -100.0
        self._spoke_until = 0.0

    # ------------------------------------------------------------------ echo gate
    @property
    def gated(self) -> bool:
        """True while microphone audio is being held back (speaker busy or its tail)."""
        return self.gate_while_speaking and (self.speaking or self.clock() < self._spoke_until)

    def release_gate(self, frames: int | None = None) -> int:
        """A player is talking over the robot: stop playback and forward the held audio.

        ``frames`` limits the forwarded audio to the most recent 250 ms blocks (the player's
        words), so the robot's own echo held before them is not sent. The rest of the
        utterance then passes straight through (see :meth:`pass_through`). Returns frames sent.
        """
        self.interrupt()
        held = list(self._held)
        self._held.clear()
        if frames is not None:
            held = held[-max(1, frames) :]
        sent = 0
        for chunk in held:
            if self._send(chunk):
                sent += 1
        self.pass_through()
        return sent

    # ------------------------------------------------------------------ addressee gate
    @property
    def holding(self) -> bool:
        """True while frames go to the utterance buffer instead of the agent."""
        return self.hold_utterances and not self._passing

    @property
    def held_seconds(self) -> float:
        with self._lock:
            return len(self._utterance) * FRAME_S

    def pass_through(self) -> int:
        """The decision is already known: forward what is held and let the rest stream live.

        Returns the frames forwarded. The gate closes again ``turn_tail_s`` after
        :meth:`utterance_ended` is called.
        """
        if not self.hold_utterances:
            return 0
        with self._lock:
            pending = list(self._utterance)
            self._utterance.clear()
            self._passing = True
            self._pass_frames_left = -1
        sent = sum(1 for _, chunk in pending if self._send(chunk))
        self.released_frames += sent
        return sent

    def utterance_ended(self) -> None:
        """The local VAD closed the utterance: a passing gate forwards the tail, then holds again."""
        with self._lock:
            if self._passing and self._pass_frames_left < 0:
                self._pass_frames_left = max(1, int(round(self.turn_tail_s / FRAME_S)))

    def release_utterance(self, until: float, *, tail_s: float | None = None) -> int:
        """Forward the held frames that arrived up to ``until`` (monotonic) plus a silent tail.

        Frames that arrived later belong to the next utterance and stay held. Returns the
        frames forwarded (the tail not counted).
        """
        pending, kept = self._split(until)
        sent = sum(1 for _, chunk in pending if self._send(chunk))
        if sent:
            tail = self.turn_tail_s if tail_s is None else tail_s
            silence = np.zeros(INPUT_BLOCK, dtype=np.int16).tobytes()
            for _ in range(int(round(tail / FRAME_S))):
                self._send(silence)
        self.released_frames += sent
        self._restore(kept)
        return sent

    def discard_utterance(self, until: float) -> int:
        """Drop the held frames that arrived up to ``until``; the agent never hears them."""
        pending, kept = self._split(until)
        self.discarded_frames += len(pending)
        self._restore(kept)
        return len(pending)

    def _restore(self, kept: list[tuple[float, bytes]]) -> None:
        """Put back the frames that arrived after ``until``, ahead of anything held since."""
        if kept:
            with self._lock:
                self._utterance.extendleft(reversed(kept))

    def _split(self, until: float) -> tuple[list[tuple[float, bytes]], list[tuple[float, bytes]]]:
        with self._lock:
            frames = list(self._utterance)
            self._utterance.clear()
        pending = [f for f in frames if f[0] <= until]
        kept = [f for f in frames if f[0] > until]
        return pending, kept

    def _send(self, chunk: bytes) -> bool:
        if self._callback is None or self.muted:
            return False
        self._callback(chunk)
        return True

    # ------------------------------------------------------------------ plumbing
    def _on_input(self, indata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        stamp = self.clock()
        frame = indata[:, 0].copy()
        for tap in self.taps:
            try:
                tap(frame)
            except Exception as exc:  # a tap must never break the uplink
                log.warning("audio tap failed: %s", exc)
        if self._callback is None or self.muted:
            return
        if self.gated:
            self._held.append(frame.tobytes())
            self.gated_frames += 1
            return
        self._held.clear()
        data = frame.tobytes()
        if not self.hold_utterances:
            self._callback(data)
            return
        with self._lock:
            if self._passing:
                forward = True
                if self._pass_frames_left > 0:
                    self._pass_frames_left -= 1
                    if self._pass_frames_left == 0:
                        self._passing = False
                        self._pass_frames_left = -1
            else:
                forward = False
                self._utterance.append((stamp, data))
                self.held_frames += 1
        if forward:
            self._callback(data)

    def _play_loop(self) -> None:
        pending = np.zeros(0, dtype=np.int16)
        while not self._stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                if pending.size:
                    self._write(pending)
                    pending = np.zeros(0, dtype=np.int16)
                self.speaking = False
                self.level_db = -100.0
                continue
            if chunk is None:
                break
            pending = np.concatenate([pending, np.frombuffer(chunk, dtype=np.int16)])
            while pending.size >= OUTPUT_BLOCK and not self._stop.is_set():
                block, pending = pending[:OUTPUT_BLOCK], pending[OUTPUT_BLOCK:]
                self._write(block)

    def _write(self, block: np.ndarray) -> None:
        if self._out is None:
            return
        self.speaking = True
        now = self.clock()
        self.last_played_at = now
        self._spoke_until = now + OUTPUT_BLOCK / SAMPLE_RATE + self.gate_tail_s
        self.level_db = rms_dbfs(block)
        try:
            self._out.write(block.reshape(-1, 1))
        except Exception as exc:
            log.warning("speaker write failed: %s", exc)


__all__ = [
    "RobotAudioInterface",
    "resolve_output_device",
    "rms_dbfs",
    "SAMPLE_RATE",
    "INPUT_BLOCK",
    "FRAME_S",
    "TURN_TAIL_S",
]
