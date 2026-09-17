"""Barge-in that tells a player from the robot's own echo, and interrupts only for the player.

Energy alone cannot tell a player from the robot's echo at the Mac microphone (both peak
around -25 dBFS), so while the robot speaks the microphone runs with a small extra margin and
every stretch of "voice" longer than ``min_ms`` is checked against the newest ``window_s``
seconds of it.

Two checks, in order of preference:

* **Voiceprint** (when voiceprints are enrolled). The speaker embedding of the tail is matched
  against the known players. A named player is talking over the robot; anything else is echo.
  This is the fast path (tens of milliseconds) and it does not care how much of the segment is
  the robot, which is what the transcript check gets wrong.
* **Transcript** (nobody enrolled, or the embedding failed). Whisper on the tail; if most of
  the words belong to what the robot is currently saying it is echo, otherwise it is a player.

Only the tail is judged, never the whole segment: while the robot speaks, the voice segment
opens on its own echo and never closes, so after a couple of seconds the segment is mostly the
robot whatever the player says - and every check came back "echo".

    barge_in = EchoAwareBargeIn(mic, transcriber, spoken_text=lambda: sentence, registry=registry)
    robot.say(clip, interrupt=barge_in)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from src.logger import get_logger
from src.speech.addressee import looks_like_echo

log = get_logger(__name__)

MIN_WORDS = 3  # fewer words cannot be judged; the echo check handles short fragments
WINDOW_S = 2.0  # how much of the voice in progress a check listens to


class EchoAwareBargeIn:
    def __init__(
        self,
        mic: Any,
        transcriber: Any,
        spoken_text: Callable[[], str],
        *,
        min_ms: int = 900,
        recheck_ms: int = 1200,
        fallback_language: str | Callable[[], str] = "",
        registry: Any = None,
        window_s: float = WINDOW_S,
        sample_rate: int = 16_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.mic = mic
        self.transcriber = transcriber
        self.spoken_text = spoken_text
        self.min_ms = min_ms
        self.recheck_ms = recheck_ms
        self.fallback_language = fallback_language
        self.registry = registry
        self.window_s = window_s
        self.sample_rate = sample_rate
        self.clock = clock
        self._checked_ms = 0  # voice length at the last check
        self.heard: str = ""  # what the player said when the interruption fired (transcript path)
        self.speaker: str = ""  # who they were (voiceprint path)
        self.score: float = 0.0
        self.checks = 0

    def __call__(self) -> bool:
        voice_ms = int(self.mic.voice_ms)
        if voice_ms < self.min_ms:
            if voice_ms == 0:
                self._checked_ms = 0
            return False
        if self._checked_ms and voice_ms - self._checked_ms < self.recheck_ms:
            return False
        audio = self.mic.current_audio()
        if audio.size == 0:
            return False
        self._checked_ms = voice_ms
        self.checks += 1
        tail = audio[-max(1, int(self.window_s * self.sample_rate)) :]
        started = self.clock()
        if self.registry is not None and self.registry.names:
            player, detail = self._by_voiceprint(tail)
        else:
            player, detail = self._by_transcript(tail)
        log.info(
            "barge-in check %d (%.1fs voice, %.1fs tail, %.2fs): %s -> %s",
            self.checks,
            voice_ms / 1000,
            tail.size / self.sample_rate,
            self.clock() - started,
            detail,
            "player" if player else "echo",
        )
        return player

    def _by_voiceprint(self, audio: np.ndarray) -> tuple[bool, str]:
        """A known player's voice over the robot: the embedding matches one of the voiceprints."""
        try:
            name, score = self.registry.identify(audio)
        except Exception as exc:
            log.warning("voiceprint check failed (%s); reading the words instead", exc)
            return self._by_transcript(audio)
        if name:
            self.speaker, self.score, self.heard = name, score, ""
            return True, f"{name} {score:.2f}"
        return False, f"no known voice ({score:.2f})"

    def _by_transcript(self, audio: np.ndarray) -> tuple[bool, str]:
        """Words that are not the robot's own: the fallback when nobody is enrolled."""
        fallback = self.fallback_language() if callable(self.fallback_language) else self.fallback_language
        result = self.transcriber.transcribe(audio, fallback_language=fallback)
        text = result.text
        if text and len(text.split()) >= MIN_WORDS and not looks_like_echo(text, self.spoken_text()):
            self.heard, self.speaker, self.score = text, "", 0.0
            return True, repr(text)
        return False, repr(text)


__all__ = ["EchoAwareBargeIn", "MIN_WORDS", "WINDOW_S"]
