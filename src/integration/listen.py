"""Listening loop: Mac microphone -> Whisper -> who / for me? -> retrieval + LLM -> voice -> robot.

    uv run python -m src.integration.listen                    # daemon on :8000 required
    uv run python -m src.integration.listen --no-robot         # gestures logged, audio on the Mac
    uv run python -m src.integration.listen --players "Ana,Bruno"   # enrol voices, then listen
    uv run python -m src.integration.listen --always-answer    # no addressee gating (Phase 2a behaviour)
    uv run python -m src.integration.listen --transcribe-only  # measure hearing alone
    uv run python -m src.integration.listen --wav clip.wav     # feed a file instead of the mic
    uv run python -m src.integration.listen --list-devices

Every heard utterance is transcribed, attributed to a player (voiceprint match, plus the
live diarizer's anonymous label when the sidecar is running), classified as addressed to the
robot or not, and logged to ``data/game_logs/addressee.jsonl``. The robot answers only when
addressed (or with ``--always-answer``). Turns are timed from the moment the person stops
talking ("ear") to the moment the robot starts talking ("mouth"). Barge-in: a voice clearly
louder than the robot's own echo for ``BARGE_IN_MIN_MS`` interrupts it.
"""

from __future__ import annotations

import argparse
import random
import signal
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import (
    BARGE_IN_MARGIN_DB,
    BARGE_IN_MIN_MS,
    DEFAULT_LANGUAGE,
    OLLAMA_MODEL,
    VAD_SILENCE_MS,
    WHISPER_MODEL,
    validate_config,
)
from src.integration.answering import think, voice
from src.llm.ollama_client import LLMError
from src.logger import get_logger
from src.robot.reachy import AGREE_MOVES, GREETING_MOVES, OOPS_MOVES, THINKING_MOVES, Robot
from src.speech.addressee import Decision, TurnLogger, decide, looks_like_echo
from src.speech.asr import ASRError, Transcriber
from src.speech.diarization import DiarizerClient
from src.speech.language import detect_language
from src.speech.microphone import (
    Microphone,
    MicrophoneError,
    Utterance,
    list_input_devices,
    read_wav,
)
from src.speech.speakers import MIN_ENROLL_S, SpeakerRegistry
from src.speech.tts import TTSError, synthesize

log = get_logger(__name__)

STAGES = ("vad_tail_s", "asr_s", "who_s", "retrieve_s", "llm_s", "tts_s", "ear_to_mouth_s", "audio_s")

ENROL_PROMPTS = {
    "pt-BR": "{name}, diga uma frase longa, de uns cinco segundos, para eu aprender a sua voz.",
    "en-US": "{name}, say a long sentence, about five seconds, so I can learn your voice.",
}
ENROL_THANKS = {"pt-BR": "Obrigado, {name}.", "en-US": "Thank you, {name}."}


@dataclass
class Session:
    """What the loop needs to remember between utterances."""

    transcriber: Transcriber
    robot: Robot
    mic: Microphone | None
    registry: SpeakerRegistry | None = None
    diarizer: DiarizerClient | None = None
    turn_log: TurnLogger | None = None
    speak: bool = True
    transcribe_only: bool = False
    barge_in: bool = True
    always_answer: bool = False
    last_answer: str = ""
    last_addressed_speaker: str = ""
    last_language: dict[str, str] = field(default_factory=dict)  # per speaker name ("" = unknown)
    robot_spoke_at: float | None = None  # monotonic
    robot_turn: bool = False
    turns: list[dict] = field(default_factory=list)


def wall_time(monotonic_t: float) -> float:
    """Convert a ``time.monotonic()`` instant to ``time.time()`` seconds."""
    return time.time() - (time.monotonic() - monotonic_t)


def who_spoke(session: Session, utterance: Utterance) -> dict[str, float | str]:
    """Voiceprint name and score for an utterance (empty name = nobody enrolled matches)."""
    who: dict[str, float | str] = {"speaker": "", "speaker_score": 0.0, "diart_label": ""}
    t0 = time.monotonic()
    if session.registry is not None and session.registry.names:
        try:
            name, score = session.registry.identify(utterance.audio)
            who["speaker"], who["speaker_score"] = name, score
        except Exception as exc:  # the loop must survive a model hiccup
            log.warning("speaker identification failed: %s", exc)
    who["who_s"] = round(time.monotonic() - t0, 3)
    return who


def diart_label(session: Session, utterance: Utterance, wait_s: float = 1.5) -> tuple[str, float]:
    """The live diarizer's anonymous label for the utterance, if the sidecar is running."""
    if session.diarizer is None or not session.diarizer.connected:
        return "", 0.0
    return session.diarizer.speaker_between(
        wall_time(utterance.started_at), wall_time(utterance.speech_ended_at), wait_s=wait_s
    )


def handle(session: Session, utterance: Utterance) -> dict | None:
    """Transcribe one utterance, decide whether it is for the robot and, if so, answer it."""
    timings: dict = {
        "heard_s": round(utterance.duration_s, 2),
        "vad_tail_s": round(utterance.ended_at - utterance.speech_ended_at, 2),
    }
    t0 = time.monotonic()
    who = who_spoke(session, utterance)
    timings.update(who)
    fallback = session.last_language.get(str(who["speaker"]), DEFAULT_LANGUAGE)
    result = session.transcriber.transcribe(
        utterance.audio, utterance.sample_rate, fallback_language=fallback
    )
    timings["asr_s"] = result.seconds
    if result.empty:
        log.info("heard nothing usable (no_speech %.2f)", result.no_speech_prob)
        return None
    language = result.language or detect_language(result.text)
    session.last_language[str(who["speaker"])] = language
    timings.update(language=language, language_confidence=result.language_confidence, text=result.text)
    log.info(
        "heard [%s] (%s %.2f): %s", who["speaker"] or "?", language, result.language_confidence, result.text
    )
    if session.transcribe_only:
        timings["ear_to_text_s"] = round(t0 + result.seconds - utterance.speech_ended_at, 2)
        return timings

    since_robot = None if session.robot_spoke_at is None else time.monotonic() - session.robot_spoke_at
    if looks_like_echo(result.text, session.last_answer):
        decision = Decision(False, "self_echo", 0.9)
    elif session.always_answer:
        decision = Decision(True, "always_answer", 1.0)
    else:
        humans = (
            len(session.registry.names) if session.registry is not None and session.registry.names else None
        )
        decision = decide(
            result.text,
            language,
            seconds_since_robot_spoke=since_robot,
            robot_turn=session.robot_turn,
            humans_present=humans,
            follow_up_ok=bool(who["speaker"]) and who["speaker"] == session.last_addressed_speaker,
        )
    timings.update(addressed=decision.addressed, reason=decision.reason)
    # The sidecar labels audio ~1.5 s after hearing it; wait for it only when staying quiet,
    # otherwise take whatever it has so far (the label is logged, not used to decide).
    label, seconds = diart_label(session, utterance, wait_s=0.0 if decision.addressed else 1.5)
    timings.update(diart_label=label, diart_overlap_s=round(seconds, 2))
    if session.turn_log is not None:
        session.turn_log.log(
            decision,
            text=result.text,
            language=language,
            speaker=who["speaker"],
            speaker_score=who["speaker_score"],
            diart_label=label,
            heard_s=timings["heard_s"],
            seconds_since_robot_spoke=None if since_robot is None else round(since_robot, 1),
            capture=str(utterance.path) if utterance.path else "",
        )
    if not decision.addressed:
        log.info("staying quiet (%s)", decision.reason)
        return timings

    session.last_addressed_speaker = str(who["speaker"])
    session.robot.emotion(random.choice(THINKING_MOVES), sound=False, block=False)
    thought = think(result.text, language)
    timings.update(thought.timings)
    session.last_answer = thought.answer
    if not session.speak:
        session.robot_spoke_at = time.monotonic()
        return timings
    clip = voice(thought)
    timings.update(thought.timings)
    mouth_at = time.monotonic()
    timings["ear_to_mouth_s"] = round(mouth_at - utterance.speech_ended_at, 2)
    log.info("ear to mouth %.2fs", timings["ear_to_mouth_s"])
    finished = say(session, clip, timings)
    session.robot_spoke_at = time.monotonic()
    if not finished:
        log.info("interrupted after %.1fs; listening", time.monotonic() - mouth_at)
    return timings


def say(session: Session, clip, timings: dict) -> bool:
    """Play a clip with barge-in monitoring; records the echo level the microphone saw."""
    mic = session.mic
    interrupt = None
    if mic is not None:
        mic.discard()
        if session.barge_in:
            mic.set_extra_margin(BARGE_IN_MARGIN_DB)
            interrupt = lambda: mic.voice_ms >= BARGE_IN_MIN_MS  # noqa: E731
    try:
        finished = session.robot.say(clip, interrupt=interrupt)
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
        if finished:
            mic.discard()  # whatever the microphone caught was the robot's own voice
    return finished


def enrol(session: Session, names: list[str], language: str = DEFAULT_LANGUAGE) -> None:
    """Ask each player for one sentence and store its voiceprint."""
    assert session.mic is not None and session.registry is not None
    prompt = ENROL_PROMPTS.get(language, ENROL_PROMPTS["en-US"])
    thanks = ENROL_THANKS.get(language, ENROL_THANKS["en-US"])
    for name in names:
        _speak_line(session, prompt.format(name=name), language)
        while True:
            utterance = session.mic.next_utterance(timeout=30.0)
            if utterance is None:
                log.warning("no sentence heard from %s; skipping", name)
                break
            if utterance.speech_s < MIN_ENROLL_S:
                log.info("too short for a voiceprint (%.1fs); say a longer sentence", utterance.speech_s)
                continue
            session.registry.enroll(name, utterance.audio, replace=True)
            session.registry.save()
            session.robot.emotion(random.choice(AGREE_MOVES), sound=False, block=False)
            _speak_line(session, thanks.format(name=name), language)
            break


def _speak_line(session: Session, text: str, language: str) -> None:
    log.info("robot: %s", text)
    if not session.speak:
        return
    try:
        say(session, synthesize(text, language), {})
        session.robot_spoke_at = time.monotonic()
    except TTSError as exc:
        log.warning("%s", exc)


def summarise(turns: list[dict]) -> str:
    """Median of every timed stage over the session, as a small text table."""
    lines = [f"{'stage':<16}{'median':>8}{'max':>8}{'n':>4}"]
    for stage in STAGES:
        values = [float(t[stage]) for t in turns if isinstance(t.get(stage), int | float)]
        if values:
            lines.append(f"{stage:<16}{statistics.median(values):>8.2f}{max(values):>8.2f}{len(values):>4}")
    answered = sum(1 for t in turns if t.get("addressed"))
    lines.append(f"utterances {len(turns)}, answered {answered}")
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
        "--always-answer", action="store_true", help="answer every utterance (no addressee rules)"
    )
    parser.add_argument(
        "--no-barge-in", action="store_true", help="never interrupt the robot while it speaks"
    )
    parser.add_argument(
        "--no-diarizer", action="store_true", help="do not connect to the live diarizer sidecar"
    )
    parser.add_argument("--players", default="", help="comma-separated names to enrol by voice at the start")
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

    registry = SpeakerRegistry.load()
    diarizer = None if args.no_diarizer else DiarizerClient().start()
    players = [n.strip() for n in args.players.split(",") if n.strip()]
    session_kwargs = dict(
        registry=registry,
        diarizer=diarizer,
        turn_log=TurnLogger(),
        speak=not args.no_speech,
        transcribe_only=args.transcribe_only,
        barge_in=not args.no_barge_in,
        always_answer=args.always_answer,
    )
    session: Session | None = None
    try:
        with Robot.connect(simulated=args.no_robot) as robot:
            if args.wav:
                session = Session(transcriber, robot, None, **session_kwargs)
                for path in args.wav:
                    _run(session, _utterance_from_wav(path))
                return 0
            with Microphone(device=args.device if args.device is not None else "") as mic:
                session = Session(transcriber, robot, mic, **session_kwargs)
                if diarizer is not None and diarizer.wait_connected(2.0):
                    log.info("live diarizer connected")
                if not args.transcribe_only:
                    robot.emotion(random.choice(GREETING_MOVES))
                if players:
                    enrol(session, players)
                log.info(
                    "listening for %s (Ctrl+C to stop); utterances end after %d ms of silence",
                    ", ".join(registry.names) or "unknown voices",
                    VAD_SILENCE_MS,
                )
                while True:
                    utterance = mic.next_utterance(timeout=0.5)
                    if utterance is not None:
                        _run(session, utterance)
    except MicrophoneError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        if diarizer is not None:
            diarizer.stop()
        if session is not None and session.turns:
            log.info("session latencies (s)\n%s", summarise(session.turns))
    return 0


def _terminate(*_: object) -> None:
    raise KeyboardInterrupt


def _run(session: Session, utterance: Utterance) -> None:
    try:
        timings = handle(session, utterance)
    except ASRError as exc:
        log.error("%s", exc)
        return
    except (LLMError, TTSError) as exc:
        log.error("%s", exc)
        session.robot.emotion(random.choice(OOPS_MOVES), block=False)
        return
    if timings is not None:
        session.turns.append(timings)
        log.info("timings %s", {k: v for k, v in timings.items() if k != "text"})


if __name__ == "__main__":
    sys.exit(main())
