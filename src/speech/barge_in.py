"""Echo-aware barge-in: interrupt the robot only for words that are not its own voice.

Energy alone cannot tell a player from the robot's echo at the Mac microphone (both peak
around -25 dBFS), so while the robot speaks the microphone runs with a small extra margin and
every stretch of "voice" longer than ``min_ms`` is transcribed. If most of the words belong to
what the robot is currently saying, it is echo and playback continues; otherwise the player
is talking over the robot and the clip is cut.

    barge_in = EchoAwareBargeIn(mic, transcriber, spoken_text=lambda: sentence_being_spoken)
    robot.say(clip, interrupt=barge_in)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.logger import get_logger
from src.speech.addressee import looks_like_echo

log = get_logger(__name__)

MIN_WORDS = 3  # fewer words cannot be judged; the echo check handles short fragments


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
        clock: Callable[[], float] = time.monotonic,
    ):
        self.mic = mic
        self.transcriber = transcriber
        self.spoken_text = spoken_text
        self.min_ms = min_ms
        self.recheck_ms = recheck_ms
        self.fallback_language = fallback_language
        self.clock = clock
        self._checked_ms = 0  # voice length at the last transcription
        self.heard: str = ""  # what the player said when the interruption fired
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
        started = self.clock()
        fallback = self.fallback_language() if callable(self.fallback_language) else self.fallback_language
        result = self.transcriber.transcribe(audio, fallback_language=fallback)
        text = result.text
        echo = looks_like_echo(text, self.spoken_text()) if text else True
        log.info(
            "barge-in check %d (%.1fs voice, %.2fs): %r -> %s",
            self.checks,
            voice_ms / 1000,
            self.clock() - started,
            text,
            "echo" if echo else "player",
        )
        if text and len(text.split()) >= MIN_WORDS and not echo:
            self.heard = text
            return True
        return False


__all__ = ["EchoAwareBargeIn", "MIN_WORDS"]
