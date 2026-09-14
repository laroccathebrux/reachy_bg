"""Continuous capture from a Mac input device, cut into utterances by a simple energy VAD.

The robot's own microphone is broken, so audio always comes from a Mac input device
(``AUDIO_INPUT_DEVICE``: empty = system default, a device index, or a case-insensitive
substring of the device name). The stream is opened at the device's native rate and resampled
to ``AUDIO_SAMPLE_RATE`` (16 kHz mono int16, what Whisper wants).

    with Microphone() as mic:
        utterance = mic.next_utterance(timeout=1.0)   # None on timeout
        utterance.audio, utterance.duration_s, utterance.path

:class:`Segmenter` holds the VAD logic and is a pure function of the frames it is fed, so it
is unit-tested without any audio device. A frame is speech when its RMS level exceeds both
an absolute floor and the adaptive noise floor plus a margin; an utterance starts after a few
consecutive speech frames (with some pre-roll) and ends after ``silence_ms`` of silence. While
the robot talks, its own voice reaches the microphone; :meth:`Segmenter.set_extra_margin`
raises the bar so only a louder, closer voice counts (barge-in).
"""

from __future__ import annotations

import math
import queue
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    AUDIO_CAPTURE_DIR,
    AUDIO_INPUT_DEVICE,
    AUDIO_SAMPLE_RATE,
    SAVE_CAPTURES,
    VAD_FRAME_MS,
    VAD_MAX_UTTERANCE_S,
    VAD_MIN_SPEECH_MS,
    VAD_NOISE_MARGIN_DB,
    VAD_PRE_ROLL_MS,
    VAD_SILENCE_MS,
    VAD_THRESHOLD_DBFS,
)
from src.logger import get_logger

log = get_logger(__name__)

_INT16_FULL_SCALE = 32768.0
_SILENCE_DB = -100.0


class MicrophoneError(RuntimeError):
    """Raised when no usable input device can be opened."""


# --------------------------------------------------------------------------- devices
@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    sample_rate: float


def list_input_devices(devices: list[dict[str, Any]] | None = None) -> list[InputDevice]:
    """Input-capable devices as seen by PortAudio (or from a pre-fetched ``query_devices`` list)."""
    if devices is None:
        import sounddevice as sd

        devices = list(sd.query_devices())
    return [
        InputDevice(i, d["name"], int(d["max_input_channels"]), float(d["default_samplerate"]))
        for i, d in enumerate(devices)
        if int(d.get("max_input_channels", 0)) > 0
    ]


def resolve_input_device(
    spec: str, devices: list[InputDevice], default_index: int | None = None
) -> InputDevice:
    """Pick the device named by ``spec`` (empty = default, digits = index, else name substring)."""
    spec = (spec or "").strip()
    if not devices:
        raise MicrophoneError("no audio input device found")
    if not spec:
        for d in devices:
            if d.index == default_index:
                return d
        return devices[0]
    if spec.isdigit():
        for d in devices:
            if d.index == int(spec):
                return d
        raise MicrophoneError(f"no input device with index {spec}")
    matches = [d for d in devices if spec.lower() in d.name.lower()]
    if not matches:
        names = ", ".join(f"{d.index}: {d.name}" for d in devices)
        raise MicrophoneError(f"no input device matching {spec!r}; available: {names}")
    return matches[0]


# --------------------------------------------------------------------------- signal helpers
def rms_dbfs(frame: np.ndarray) -> float:
    """RMS level of an int16 (or float in [-1, 1]) frame in dB relative to full scale."""
    if frame.size == 0:
        return _SILENCE_DB
    samples = frame.astype(np.float64)
    if frame.dtype.kind in "iu":
        samples /= _INT16_FULL_SCALE
    rms = math.sqrt(float(np.mean(samples * samples)))
    return 20.0 * math.log10(rms) if rms > 0 else _SILENCE_DB


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono int16 audio; integer ratios use a box filter plus decimation."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    if src_rate % dst_rate == 0:
        ratio = src_rate // dst_rate
        usable = (audio.size // ratio) * ratio
        return audio[:usable].reshape(-1, ratio).mean(axis=1).astype(np.int16)
    positions = np.arange(0, audio.size, src_rate / dst_rate)
    return np.interp(positions, np.arange(audio.size), audio.astype(np.float64)).astype(np.int16)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.astype(np.int16).tobytes())
    return path


def read_wav(path: Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Load a WAV as mono int16 at ``sample_rate`` (16-bit PCM input only)."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise MicrophoneError(f"{path}: only 16-bit PCM WAV is supported")
        frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if wav.getnchannels() > 1:
            frames = frames.reshape(-1, wav.getnchannels()).mean(axis=1).astype(np.int16)
        return resample(frames, wav.getframerate(), sample_rate)


# --------------------------------------------------------------------------- VAD
@dataclass
class Utterance:
    """One stretch of speech cut from the stream (mono int16)."""

    audio: np.ndarray
    sample_rate: int
    started_at: float  # clock time of the first speech frame
    speech_ended_at: float  # clock time of the last speech frame
    ended_at: float  # clock time when the end was decided (after the silence tail)
    peak_db: float
    path: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.audio.size / self.sample_rate

    @property
    def speech_s(self) -> float:
        return max(0.0, self.speech_ended_at - self.started_at)


class Segmenter:
    """Energy VAD that turns a stream of fixed-size frames into :class:`Utterance` objects.

    Feed it int16 frames of ``frame_ms`` with :meth:`feed`; it returns an utterance when one
    is complete. ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        *,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        frame_ms: int = VAD_FRAME_MS,
        silence_ms: int = VAD_SILENCE_MS,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        pre_roll_ms: int = VAD_PRE_ROLL_MS,
        max_utterance_s: float = VAD_MAX_UTTERANCE_S,
        threshold_dbfs: float = VAD_THRESHOLD_DBFS,
        noise_margin_db: float = VAD_NOISE_MARGIN_DB,
        start_frames: int = 3,
        noise_window_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_s = max_utterance_s
        self.threshold_dbfs = threshold_dbfs
        self.noise_margin_db = noise_margin_db
        self.start_frames = start_frames
        self.clock = clock
        self.extra_margin_db = 0.0
        self.noise_floor_db: float | None = None
        # Levels of the last ``noise_window_s`` seconds; the floor is their 10th percentile, so
        # speech (never the whole window) cannot drag it up, while a fan that starts is
        # learned within the window.
        self._levels: deque[float] = deque(maxlen=max(10, int(noise_window_s * 1000 / frame_ms)))
        self.level_db = _SILENCE_DB  # last frame
        self.max_level_db = _SILENCE_DB  # since the last reset (e.g. the robot's echo peak)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, pre_roll_ms // frame_ms))
        self._reset_state()

    def _reset_state(self) -> None:
        self.in_speech = False
        self._buffer: list[np.ndarray] = []
        self._run_frames = 0  # consecutive speech frames while idle
        self._silence_ms = 0
        self._started_at = 0.0
        self._last_speech_at = 0.0
        self._peak_db = _SILENCE_DB

    # ------------------------------------------------------------------ public state
    @property
    def threshold_db(self) -> float:
        """Current speech threshold: the absolute floor or the noise floor plus margins."""
        floor = _SILENCE_DB if self.noise_floor_db is None else self.noise_floor_db
        return max(self.threshold_dbfs, floor + self.noise_margin_db + self.extra_margin_db)

    @property
    def voice_ms(self) -> int:
        """How long the current voice has been running (0 when nobody is talking)."""
        if self.in_speech:
            return int((self._last_speech_at - self._started_at) * 1000) + self.frame_ms
        return self._run_frames * self.frame_ms

    def set_extra_margin(self, db: float) -> None:
        """Raise (or reset) the speech threshold, e.g. while the robot's own voice is playing."""
        self.extra_margin_db = max(0.0, db)

    def reset(self) -> None:
        """Drop any speech in progress (used to discard the robot's own echo)."""
        self._reset_state()
        self._pre_roll.clear()
        self.max_level_db = _SILENCE_DB

    # ------------------------------------------------------------------ feeding
    def feed(self, frame: np.ndarray) -> Utterance | None:
        now = self.clock()
        level = rms_dbfs(frame)
        self.level_db = level
        self.max_level_db = max(self.max_level_db, level)
        # The floor freezes while a barge-in margin is active (the robot's own voice would
        # otherwise be learned as "noise").
        if self.extra_margin_db == 0.0 or self.noise_floor_db is None:
            self._levels.append(level)
            self.noise_floor_db = float(np.percentile(self._levels, 10))
        is_speech = level > self.threshold_db

        if not self.in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                self._run_frames += 1
                if self._run_frames >= self.start_frames:
                    self._begin(now, level)
            else:
                self._run_frames = 0
            return None

        self._buffer.append(frame)
        if is_speech:
            self._silence_ms = 0
            self._last_speech_at = now
            self._peak_db = max(self._peak_db, level)
        else:
            self._silence_ms += self.frame_ms
        if self._silence_ms >= self.silence_ms:
            return self._finish(now)
        if len(self._buffer) * self.frame_ms >= self.max_utterance_s * 1000:
            return self._finish(now, forced=True)
        return None

    def flush(self) -> Utterance | None:
        """End the stream: return the utterance in progress, if any."""
        return self._finish(self.clock(), forced=True) if self.in_speech else None

    def _begin(self, now: float, level: float) -> None:
        self.in_speech = True
        self._buffer = list(self._pre_roll)
        self._pre_roll.clear()
        run_s = self._run_frames * self.frame_ms / 1000
        self._started_at = now - run_s + self.frame_ms / 1000
        self._last_speech_at = now
        self._silence_ms = 0
        self._peak_db = level
        self._run_frames = 0

    def _finish(self, now: float, *, forced: bool = False) -> Utterance | None:
        audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.int16)
        utterance = Utterance(
            audio=audio,
            sample_rate=self.sample_rate,
            started_at=self._started_at,
            speech_ended_at=self._last_speech_at,
            ended_at=now,
            peak_db=self._peak_db,
            meta={"forced": forced, "threshold_db": round(self.threshold_db, 1)},
        )
        self._reset_state()
        if utterance.speech_s * 1000 + self.frame_ms < self.min_speech_ms:
            return None
        return utterance


# --------------------------------------------------------------------------- capture
class Microphone:
    """Background capture thread feeding a :class:`Segmenter`; utterances come out of a queue."""

    def __init__(
        self,
        device: str = AUDIO_INPUT_DEVICE,
        *,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        segmenter: Segmenter | None = None,
        save_dir: Path | None = AUDIO_CAPTURE_DIR if SAVE_CAPTURES else None,
    ):
        self.device_spec = device
        self.sample_rate = sample_rate
        self.segmenter = segmenter or Segmenter(sample_rate=sample_rate)
        self.save_dir = save_dir
        self.device: InputDevice | None = None
        self.device_rate = sample_rate
        self._stream: Any | None = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._utterances: queue.Queue[Utterance] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()
        self._pending = np.zeros(0, dtype=np.int16)

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> Microphone:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        import sounddevice as sd

        devices = list_input_devices()
        default_in = (
            sd.default.device[0] if isinstance(sd.default.device, list | tuple) else sd.default.device
        )
        self.device = resolve_input_device(self.device_spec, devices, default_in)
        self.device_rate = self._pick_rate(sd, self.device)
        block = self.device_rate * self.segmenter.frame_ms // 1000
        self._stream = sd.InputStream(
            device=self.device.index,
            channels=1,
            samplerate=self.device_rate,
            dtype="int16",
            blocksize=block,
            callback=self._on_audio,
        )
        self._stopping.clear()
        self._worker = threading.Thread(target=self._run, name="microphone", daemon=True)
        self._worker.start()
        self._stream.start()
        log.info(
            "listening on %r (%d Hz -> %d Hz, %d ms frames)",
            self.device.name,
            self.device_rate,
            self.sample_rate,
            self.segmenter.frame_ms,
        )

    def _pick_rate(self, sd: Any, device: InputDevice) -> int:
        try:
            sd.check_input_settings(
                device=device.index, channels=1, samplerate=self.sample_rate, dtype="int16"
            )
            return self.sample_rate
        except Exception:
            return int(device.sample_rate)

    def stop(self) -> None:
        self._stopping.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.warning("closing the input stream failed: %s", exc)
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    # ------------------------------------------------------------------ stream plumbing
    def _on_audio(self, indata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        if status:
            log.debug("input status: %s", status)
        self._frames.put(indata[:, 0].copy())

    def _run(self) -> None:
        frame_samples = self.segmenter.frame_samples
        while not self._stopping.is_set():
            try:
                raw = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            audio = resample(raw, self.device_rate, self.sample_rate)
            self._pending = np.concatenate([self._pending, audio]) if self._pending.size else audio
            while self._pending.size >= frame_samples:
                frame, self._pending = self._pending[:frame_samples], self._pending[frame_samples:]
                with self._lock:
                    utterance = self.segmenter.feed(frame)
                if utterance is not None:
                    self._emit(utterance)

    def _emit(self, utterance: Utterance) -> None:
        if self.save_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            try:
                utterance.path = write_wav(self.save_dir / f"{stamp}.wav", utterance.audio, self.sample_rate)
            except OSError as exc:
                log.warning("could not save the capture: %s", exc)
        log.info(
            "utterance %.1fs (speech %.1fs, peak %.0f dBFS, threshold %.0f dBFS)",
            utterance.duration_s,
            utterance.speech_s,
            utterance.peak_db,
            utterance.meta.get("threshold_db", 0),
        )
        self._utterances.put(utterance)

    # ------------------------------------------------------------------ consumer API
    def next_utterance(self, timeout: float | None = None) -> Utterance | None:
        try:
            return self._utterances.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def voice_ms(self) -> int:
        with self._lock:
            return self.segmenter.voice_ms

    @property
    def noise_floor_db(self) -> float | None:
        with self._lock:
            return self.segmenter.noise_floor_db

    def levels(self) -> dict[str, float]:
        """Current threshold, noise floor and the loudest frame since the last ``discard``."""
        with self._lock:
            seg = self.segmenter
            return {
                "threshold_db": round(seg.threshold_db, 1),
                "noise_floor_db": round(seg.noise_floor_db or _SILENCE_DB, 1),
                "max_level_db": round(seg.max_level_db, 1),
            }

    def set_extra_margin(self, db: float) -> None:
        with self._lock:
            self.segmenter.set_extra_margin(db)

    def discard(self) -> None:
        """Drop speech in progress and any queued utterance (e.g. the robot's own echo)."""
        with self._lock:
            self.segmenter.reset()
        while True:
            try:
                self._utterances.get_nowait()
            except queue.Empty:
                break


__all__ = [
    "InputDevice",
    "Microphone",
    "MicrophoneError",
    "Segmenter",
    "Utterance",
    "list_input_devices",
    "resolve_input_device",
    "resample",
    "rms_dbfs",
    "read_wav",
    "write_wav",
]
