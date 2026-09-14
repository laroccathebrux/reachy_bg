"""Listening loop: Mac microphone -> Whisper -> retrieval + LLM -> native voice -> robot.

    uv run python -m src.integration.listen                  # daemon on :8000 required
    uv run python -m src.integration.listen --no-robot       # gestures logged, audio on the Mac
    uv run python -m src.integration.listen --transcribe-only  # measure hearing alone
    uv run python -m src.integration.listen --wav clip.wav   # feed a file instead of the mic
    uv run python -m src.integration.listen --list-devices

Every turn is timed from the moment the person stops talking ("ear") to the moment the
robot starts talking ("mouth"): VAD tail + ASR + retrieval + LLM + TTS. Barge-in: a voice
that is clearly louder than the robot's own echo for ``BARGE_IN_MIN_MS`` interrupts it.
"""

from __future__ import annotations

import argparse
import random
import signal
import statistics
import sys
import time
from pathlib import Path

from src.config import (
    BARGE_IN_MARGIN_DB,
    BARGE_IN_MIN_MS,
    OLLAMA_MODEL,
    VAD_SILENCE_MS,
    WHISPER_MODEL,
    validate_config,
)
from src.integration.answering import think, voice
from src.llm.ollama_client import LLMError
from src.logger import get_logger
from src.robot.reachy import GREETING_MOVES, OOPS_MOVES, THINKING_MOVES, Robot
from src.speech.asr import ASRError, Transcriber
from src.speech.language import detect_language
from src.speech.microphone import (
    Microphone,
    MicrophoneError,
    Utterance,
    list_input_devices,
    read_wav,
)
from src.speech.tts import TTSError

log = get_logger(__name__)

STAGES = ("vad_tail_s", "asr_s", "retrieve_s", "llm_s", "tts_s", "ear_to_mouth_s", "audio_s")


class Turn:
    """One heard utterance handled end to end, with its timings."""

    def __init__(self, utterance: Utterance):
        self.utterance = utterance
        self.timings: dict[str, float | str | bool] = {
            "heard_s": round(utterance.duration_s, 2),
            "vad_tail_s": round(utterance.ended_at - utterance.speech_ended_at, 2),
        }


def handle(
    turn: Turn,
    transcriber: Transcriber,
    robot: Robot,
    mic: Microphone | None,
    *,
    speak: bool = True,
    transcribe_only: bool = False,
    barge_in: bool = True,
) -> dict[str, float | str | bool] | None:
    """Transcribe one utterance and, unless ``transcribe_only``, answer it out loud."""
    utterance, timings = turn.utterance, turn.timings
    t0 = time.monotonic()
    result = transcriber.transcribe(utterance.audio, utterance.sample_rate)
    timings["asr_s"] = result.seconds
    if result.empty:
        log.info("heard nothing usable (no_speech %.2f)", result.no_speech_prob)
        return None
    language = result.language or detect_language(result.text)
    timings["language"] = language
    timings["language_confidence"] = result.language_confidence
    timings["text"] = result.text
    log.info("heard (%s %.2f): %s", language, result.language_confidence, result.text)
    if transcribe_only:
        timings["ear_to_text_s"] = round(t0 + result.seconds - utterance.speech_ended_at, 2)
        return timings

    robot.emotion(random.choice(THINKING_MOVES), sound=False, block=False)
    thought = think(result.text, language)
    timings.update(thought.timings)
    if not speak:
        return timings
    clip = voice(thought)
    timings.update(thought.timings)
    mouth_at = time.monotonic()
    timings["ear_to_mouth_s"] = round(mouth_at - utterance.speech_ended_at, 2)
    log.info("ear to mouth %.2fs", timings["ear_to_mouth_s"])

    interrupt = None
    if mic is not None:
        mic.discard()
        if barge_in:
            mic.set_extra_margin(BARGE_IN_MARGIN_DB)
            interrupt = lambda: mic.voice_ms >= BARGE_IN_MIN_MS  # noqa: E731
    try:
        finished = robot.say(clip, interrupt=interrupt)
    finally:
        if mic is not None:
            levels = mic.levels()  # read while the barge-in margin is still applied
            mic.set_extra_margin(0.0)
    timings["interrupted"] = not finished
    if mic is not None:
        timings["echo_peak_db"] = levels["max_level_db"]
        log.info(
            "while speaking: mic peak %.0f dBFS, barge-in threshold %.0f dBFS, noise floor %.0f dBFS",
            levels["max_level_db"],
            levels["threshold_db"],
            levels["noise_floor_db"],
        )
    if not finished:
        log.info("interrupted after %.1fs; listening", time.monotonic() - mouth_at)
    elif mic is not None:
        mic.discard()  # whatever the microphone caught was the robot's own voice
    return timings


def summarise(turns: list[dict[str, float | str | bool]]) -> str:
    """Median of every timed stage over the session, as a small text table."""
    lines = [f"{'stage':<16}{'median':>8}{'max':>8}{'n':>4}"]
    for stage in STAGES:
        values = [float(t[stage]) for t in turns if isinstance(t.get(stage), int | float)]
        if values:
            lines.append(f"{stage:<16}{statistics.median(values):>8.2f}{max(values):>8.2f}{len(values):>4}")
    return "\n".join(lines)


def _utterance_from_wav(path: Path) -> Utterance:
    audio = read_wav(path)
    now = time.monotonic()
    duration = audio.size / 16_000
    return Utterance(
        audio=audio,
        sample_rate=16_000,
        started_at=now - duration,
        speech_ended_at=now,
        ended_at=now,
        peak_db=0.0,
        path=path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-robot", action="store_true", help="no daemon: log gestures, play audio on the Mac"
    )
    parser.add_argument("--no-speech", action="store_true", help="answer in the log only (no TTS)")
    parser.add_argument("--transcribe-only", action="store_true", help="print transcripts, never answer")
    parser.add_argument(
        "--no-barge-in", action="store_true", help="never interrupt the robot while it speaks"
    )
    parser.add_argument("--device", default=None, help="input device name substring or index")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    parser.add_argument("--wav", action="append", type=Path, help="WAV file(s) to treat as heard speech")
    args = parser.parse_args(argv)

    if args.list_devices:
        for d in list_input_devices():
            print(f"{d.index:>3}  {d.name}  ({d.channels} ch, {d.sample_rate:.0f} Hz)")
        return 0

    problems = validate_config()
    for problem in problems:
        log.warning("config: %s", problem)
    if problems and not (args.no_speech or args.transcribe_only):
        log.error("fix the configuration above or pass --no-speech")
        return 1
    log.info("models: %s (LLM), %s (ASR)", OLLAMA_MODEL, WHISPER_MODEL)

    transcriber = Transcriber()
    try:
        log.info("whisper warm-up: %.1fs", transcriber.warm_up())
    except ASRError as exc:
        log.error("%s", exc)
        return 1

    # A background run (nohup, `&`) is stopped with SIGTERM; treat it like Ctrl+C so the robot
    # is left quiet and the latency summary is printed.
    signal.signal(signal.SIGTERM, _terminate)

    turns: list[dict[str, float | str | bool]] = []
    speak = not args.no_speech
    kwargs = {"speak": speak, "transcribe_only": args.transcribe_only, "barge_in": not args.no_barge_in}
    try:
        with Robot.connect(simulated=args.no_robot) as robot:
            if args.wav:
                for path in args.wav:
                    _run(Turn(_utterance_from_wav(path)), transcriber, robot, None, turns, **kwargs)
                return 0
            with Microphone(device=args.device if args.device is not None else "") as mic:
                if not args.transcribe_only:
                    robot.emotion(random.choice(GREETING_MOVES))
                log.info(
                    "say something (Ctrl+C to stop); utterances end after %d ms of silence", VAD_SILENCE_MS
                )
                while True:
                    utterance = mic.next_utterance(timeout=0.5)
                    if utterance is None:
                        continue
                    _run(Turn(utterance), transcriber, robot, mic, turns, **kwargs)
    except MicrophoneError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        if turns:
            log.info("session latencies (s)\n%s", summarise(turns))
    return 0


def _terminate(*_: object) -> None:
    raise KeyboardInterrupt


def _run(
    turn: Turn, transcriber: Transcriber, robot: Robot, mic: Microphone | None, turns: list, **kwargs
) -> None:
    try:
        timings = handle(turn, transcriber, robot, mic, **kwargs)
    except ASRError as exc:
        log.error("%s", exc)
        return
    except (LLMError, TTSError) as exc:
        log.error("%s", exc)
        robot.emotion(random.choice(OOPS_MOVES), block=False)
        return
    if timings is not None:
        turns.append(timings)
        log.info("timings %s", {k: v for k, v in timings.items() if k != "text"})


if __name__ == "__main__":
    sys.exit(main())
