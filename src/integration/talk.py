"""Table conversation on an ElevenLabs agent, with the robot's ears, memory and body kept local.

    uv run python -m src.integration.talk                     # daemon on :8000, Mac mic, robot speaker
    uv run python -m src.integration.talk --no-robot          # gestures logged, robot speaker still used
    uv run python -m src.integration.talk --players "Ana,Bruno"   # enrol voices first (local)
    uv run python -m src.integration.talk --device "MacBook Pro Microphone"

The agent does ASR, LLM, TTS and turn-taking in the cloud (about half a second per turn)
and calls back into this process for rules and knowledge (Qdrant). Locally, the same
microphone audio feeds the voice activity detector, the voiceprints and the live diarizer,
so every heard utterance is still attributed to a player and judged by the addressee rules;
those decisions are logged as *shadow* decisions next to what the agent actually did, which
is the turn-taking dataset. Everything heard and said goes to data/game_logs/conversation.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_LANGUAGE,
    ELEVENLABS_API_KEY,
    GAME_LOG_DIR,
    HEAD_SWAY,
    validate_config,
)
from src.logger import get_logger
from src.robot.reachy import AGREE_MOVES, GREETING_MOVES, Robot
from src.robot.sway import HeadSway
from src.speech.addressee import Decision, TurnLogger, decide, looks_like_echo
from src.speech.agent_audio import RobotAudioInterface
from src.speech.asr import Transcriber
from src.speech.barge_in import EchoAwareBargeIn
from src.speech.diarization import DiarizerClient
from src.speech.eleven_agent import client_tools, ensure_agent, session_override
from src.speech.language import detect_language
from src.speech.microphone import Segmenter, Utterance, write_wav
from src.speech.speakers import MIN_ENROLL_S, SpeakerRegistry
from src.speech.tts import TTSError, synthesize

log = get_logger(__name__)

ENROL_PROMPTS = {
    "pt-BR": "{name}, diga uma frase longa, de uns cinco segundos, para eu aprender a sua voz.",
    "en-US": "{name}, say a long sentence, about five seconds, so I can learn your voice.",
}
ENROL_THANKS = {"pt-BR": "Obrigado, {name}.", "en-US": "Thank you, {name}."}


def wall_time(monotonic_t: float) -> float:
    return time.time() - (time.monotonic() - monotonic_t)


class Diary:
    """One JSON line per event: heard, said, tool, interrupted, latency."""

    def __init__(self, path: Path | None = None):
        self.path = path or GAME_LOG_DIR / "conversation.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, kind: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "kind": kind, **fields}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class LocalEar:
    """Local VAD + voiceprints on the microphone tap: who said the last thing, and when."""

    segmenter: Segmenter
    registry: SpeakerRegistry | None
    diarizer: DiarizerClient | None
    save_dir: Path | None
    recent: list[tuple[Utterance, str, float]] = field(default_factory=list)  # (utterance, name, score)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))

    def feed(self, frame: np.ndarray) -> None:
        n = self.segmenter.frame_samples
        with self.lock:
            self.pending = np.concatenate([self.pending, frame]) if self.pending.size else frame
            while self.pending.size >= n:
                chunk, self.pending = self.pending[:n], self.pending[n:]
                utterance = self.segmenter.feed(chunk)
                if utterance is not None:
                    self._heard(utterance)

    def _heard(self, utterance: Utterance) -> None:
        name, score = "", 0.0
        if self.registry is not None and self.registry.names:
            try:
                name, score = self.registry.identify(utterance.audio)
            except Exception as exc:
                log.warning("speaker identification failed: %s", exc)
        if self.save_dir is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            try:
                utterance.path = write_wav(
                    self.save_dir / f"{stamp}_{int(time.time() * 1000) % 1000:03d}.wav",
                    utterance.audio,
                    16_000,
                )
            except OSError as exc:
                log.warning("could not save the capture: %s", exc)
        self.recent.append((utterance, name, score))
        self.recent = self.recent[-20:]

    # The barge-in checker reads the speech in progress through these two.
    @property
    def voice_ms(self) -> int:
        with self.lock:
            return self.segmenter.voice_ms

    def current_audio(self) -> np.ndarray:
        with self.lock:
            return self.segmenter.current_audio()

    def last_utterance(self, max_age_s: float = 12.0) -> tuple[Utterance, str, float] | None:
        with self.lock:
            candidates = [r for r in self.recent if time.monotonic() - r[0].ended_at <= max_age_s]
        return candidates[-1] if candidates else None

    def wait_utterance(self, timeout: float) -> Utterance | None:
        deadline = time.monotonic() + timeout
        seen = len(self.recent)
        while time.monotonic() < deadline:
            if len(self.recent) > seen:
                return self.recent[-1][0]
            time.sleep(0.1)
        return None


class TableConversation:
    """SDK conversation that also reports the raw events the SDK ignores (system tool responses)."""

    def __init__(self, *args: Any, on_raw: Any = None, **kwargs: Any):
        from elevenlabs.conversational_ai.conversation import Conversation

        self._impl = Conversation(*args, **kwargs)
        self._on_raw = on_raw
        original = self._impl._handle_message

        def handle(message: dict, ws: Any) -> None:
            if self._on_raw is not None:
                try:
                    self._on_raw(message)
                except Exception as exc:
                    log.warning("raw event hook failed: %s", exc)
            original(message, ws)

        self._impl._handle_message = handle  # type: ignore[method-assign]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


class Table:
    """Callbacks of the agent conversation, wired to the local ear, the diary and the body."""

    def __init__(
        self, robot: Robot, ear: LocalEar, diary: Diary, turn_log: TurnLogger, audio: RobotAudioInterface
    ):
        self.robot = robot
        self.ear = ear
        self.diary = diary
        self.turn_log = turn_log
        self.audio = audio
        self.robot_spoke_at: float | None = None
        self.last_addressed: str = ""
        self.conversation: Any = None
        self.turns = 0
        self.interruptions = 0
        self.responses: list[str] = []  # last few agent lines: a filler and its answer overlap in time
        self.barge_ins = 0
        self.transcriber: Transcriber | None = None
        self.voice_language = DEFAULT_LANGUAGE  # the language (and native voice) of the current session
        self.restart: Any = None  # set by main: restart(language, text)
        self._switching = False

    def on_user_transcript(self, text: str) -> None:
        heard = self.ear.last_utterance()
        name, score, label = "", 0.0, ""
        if heard is not None:
            utterance, name, score = heard
            if self.ear.diarizer is not None and self.ear.diarizer.connected:
                label, _ = self.ear.diarizer.speaker_between(
                    wall_time(utterance.started_at), wall_time(utterance.speech_ended_at)
                )
        language = detect_language(text)
        since = None if self.robot_spoke_at is None else round(time.monotonic() - self.robot_spoke_at, 1)
        humans = (
            len(self.ear.registry.names)
            if self.ear.registry is not None and self.ear.registry.names
            else None
        )
        if looks_like_echo(text, self.spoken_recently):
            decision = Decision(False, "self_echo", 0.9)
        else:
            decision = decide(
                text,
                language,
                seconds_since_robot_spoke=since,
                humans_present=humans,
                follow_up_ok=bool(name) and name == self.last_addressed,
            )
        self.turn_log.log(
            decision,
            text=text,
            language=language,
            speaker=name,
            speaker_score=score,
            diart_label=label,
            seconds_since_robot_spoke=since,
            shadow=True,
            capture=str(heard[0].path) if heard is not None and heard[0].path else "",
        )
        self.diary.write(
            "heard",
            text=text,
            speaker=name,
            score=score,
            diart=label,
            shadow_addressed=decision.addressed,
            reason=decision.reason,
        )
        self.last_addressed = name
        # Cheap, immediate hint from the transcript text; the Whisper check below catches the
        # cases where the agent's ASR (pinned to the current language) mangled the words.
        if language != self.voice_language and len(text.split()) >= 3:
            self.switch_language(language, text, "transcript")
        elif heard is not None and self.transcriber is not None:
            threading.Thread(
                target=self._check_language, args=(heard[0],), name="language-watch", daemon=True
            ).start()
        log.info(
            "heard [%s %.2f%s]: %s  (shadow: %s, %s)",
            name or "?",
            score,
            f" {label}" if label else "",
            text,
            "answer" if decision.addressed else "quiet",
            decision.reason,
        )

    def _check_language(self, utterance: Utterance) -> None:
        """The agent's ASR mangles Portuguese once the session is English; our Whisper catches it."""
        try:
            result = self.transcriber.transcribe(utterance.audio, utterance.sample_rate)  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("language check failed: %s", exc)
            return
        if not result.language or result.language_confidence < 0.8 or result.language == self.voice_language:
            return
        self.switch_language(result.language, result.text, f"whisper {result.language_confidence:.2f}")

    def switch_language(self, language: str, text: str, source: str) -> None:
        """Restart the agent session with the other native voice and re-ask ``text`` there.

        Runs on its own thread: ending a session from inside its receive thread would block.
        """
        if self.restart is None or self._switching:
            return
        self._switching = True
        log.info(
            "player speaks %s (%s) but the voice is %s; restarting the session",
            language,
            source,
            self.voice_language,
        )
        self.diary.write("language_switch", language=language, source=source, text=text)
        self.audio.interrupt()

        def run() -> None:
            try:
                self.restart(language, text)
            except Exception as exc:
                log.warning("language switch failed: %s", exc)
            finally:
                self._switching = False

        threading.Thread(target=run, name="language-switch", daemon=True).start()

    def on_raw_event(self, message: dict) -> None:
        """Keep the raw system tool events in the diary (the SDK does not surface them)."""
        if message.get("type") != "agent_tool_response":
            return
        event = message.get("agent_tool_response_event") or {}
        self.diary.write("agent_tool", event={k: str(v)[:80] for k, v in event.items()})

    @property
    def spoken_recently(self) -> str:
        return " ".join(self.responses)

    def on_agent_response(self, text: str) -> None:
        self.turns += 1
        self.robot_spoke_at = time.monotonic()
        self.responses = (self.responses + [text])[-3:]
        self.diary.write("said", text=text)
        log.info("robot: %s", text)

    def watch_for_barge_in(self, transcriber: Transcriber, stop: threading.Event) -> None:
        """While the robot speaks, transcribe any voice the gate is holding back; a player's words release it."""
        checker: EchoAwareBargeIn | None = None
        while not stop.is_set():
            time.sleep(0.1)
            if not self.audio.gated:
                checker = None
                continue
            if checker is None:
                checker = EchoAwareBargeIn(
                    self.ear,
                    transcriber,
                    lambda: self.spoken_recently,
                    min_ms=900,
                    fallback_language=lambda: detect_language(self.spoken_recently),
                )
            if checker():
                # Forward only the stretch where the voice was heard, not the robot's echo before it.
                sent = self.audio.release_gate(frames=int(self.ear.voice_ms / 250) + 2)
                self.barge_ins += 1
                self.diary.write("barge_in", heard=checker.heard, frames_forwarded=sent)
                log.info("player talked over the robot (%r); forwarded %d held frames", checker.heard, sent)
                checker = None

    def on_agent_correction(self, original: str, corrected: str) -> None:
        self.diary.write("said_correction", original=original, corrected=corrected)
        log.info("robot (cut short): %s", corrected)

    def on_interruption(self) -> None:
        self.interruptions += 1
        self.audio.interrupt()
        self.diary.write("interrupted")
        log.info("interrupted by the table; listening")

    def on_latency(self, ms: int) -> None:
        self.diary.write("latency", ms=ms)
        log.info("turn latency %d ms", ms)

    def on_tool(self, name: str, parameters: dict, result: dict) -> None:
        self.diary.write("tool", name=name, parameters=parameters, passages=result.get("passages", "")[:400])
        log.info("tool %s %s -> %d chars", name, parameters, len(result.get("passages", "")))


def enrol(
    robot: Robot, ear: LocalEar, names: list[str], language: str = DEFAULT_LANGUAGE, *, speak: bool = True
) -> None:
    assert ear.registry is not None
    for name in names:
        _say_line(
            robot, ENROL_PROMPTS.get(language, ENROL_PROMPTS["en-US"]).format(name=name), language, speak
        )
        while True:
            utterance = ear.wait_utterance(30.0)
            if utterance is None:
                log.warning("no sentence heard from %s; skipping", name)
                break
            if utterance.speech_s < MIN_ENROLL_S:
                log.info("too short for a voiceprint (%.1fs); say a longer sentence", utterance.speech_s)
                continue
            ear.registry.enroll(name, utterance.audio, replace=True)
            ear.registry.save()
            robot.emotion(random.choice(AGREE_MOVES), sound=False, block=False)
            _say_line(
                robot, ENROL_THANKS.get(language, ENROL_THANKS["en-US"]).format(name=name), language, speak
            )
            break


def _say_line(robot: Robot, text: str, language: str, speak: bool) -> None:
    log.info("robot: %s", text)
    if not speak:
        return
    try:
        robot.say(synthesize(text, language))
    except TTSError as exc:
        log.warning("%s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-robot", action="store_true", help="no daemon: gestures logged")
    parser.add_argument("--device", default=None, help="input device name substring or index")
    parser.add_argument("--output-device", default=None, help="output device name substring")
    parser.add_argument("--players", default="", help="comma-separated names to enrol by voice at the start")
    parser.add_argument("--no-diarizer", action="store_true")
    parser.add_argument("--no-sway", action="store_true", help="do not move the head with the voice")
    args = parser.parse_args(argv)

    problems = validate_config()
    for problem in problems:
        log.warning("config: %s", problem)
    if not ELEVENLABS_API_KEY:
        log.error("ELEVENLABS_API_KEY is empty")
        return 1
    signal.signal(signal.SIGTERM, _terminate)

    from elevenlabs import ElevenLabs

    from src.config import AUDIO_CAPTURE_DIR, AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE, SAVE_CAPTURES

    registry = SpeakerRegistry.load()
    if registry.names:
        log.info("speaker model warm-up: %.1fs", registry.warm_up())
    transcriber = Transcriber()
    log.info("whisper warm-up (barge-in checks): %.1fs", transcriber.warm_up())
    diarizer = None if args.no_diarizer else DiarizerClient().start()
    ear = LocalEar(Segmenter(), registry, diarizer, AUDIO_CAPTURE_DIR if SAVE_CAPTURES else None)
    audio = RobotAudioInterface(
        args.device if args.device is not None else AUDIO_INPUT_DEVICE,
        args.output_device if args.output_device is not None else AUDIO_OUTPUT_DEVICE,
    )
    audio.taps.append(ear.feed)
    diary = Diary()
    players = [n.strip() for n in args.players.split(",") if n.strip()]

    agent_id = ensure_agent()
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    conversation = None
    sway = None
    try:
        with Robot.connect(simulated=args.no_robot) as robot:
            table = Table(robot, ear, diary, TurnLogger(), audio)
            table.transcriber = transcriber
            from elevenlabs.conversational_ai.conversation import ConversationInitiationData

            def open_session(language: str) -> Any:
                conv = TableConversation(
                    client,
                    agent_id,
                    requires_auth=True,
                    audio_interface=audio,
                    client_tools=client_tools(on_call=table.on_tool),  # a fresh loop per session
                    config=ConversationInitiationData(
                        conversation_config_override=session_override(language)
                    ),
                    callback_agent_response=table.on_agent_response,
                    callback_agent_response_correction=table.on_agent_correction,
                    callback_user_transcript=table.on_user_transcript,
                    callback_latency_measurement=table.on_latency,
                    on_raw=table.on_raw_event,
                )
                table.conversation = conv
                table.voice_language = language
                return conv

            def restart(language: str, text: str) -> None:
                nonlocal conversation
                _end_session(conversation)
                conversation = open_session(language)
                conversation.start_session()
                if not _wait_connected(conversation, 8.0):
                    log.warning("the new session did not connect in time")
                    return
                recent = "\n".join(f"- {r}" for r in table.responses[-3:])
                conversation.send_contextual_update(
                    "The voice session was restarted to switch language. Do not greet or introduce yourself; "
                    f"continue the conversation. Your last lines were:\n{recent}"
                )
                conversation.send_user_message(text)
                diary.write("session_restart", language=language)
                log.info("session restarted in %s; re-asking: %s", language, text)

            table.restart = restart
            conversation = open_session(DEFAULT_LANGUAGE)
            robot.emotion(random.choice(GREETING_MOVES))
            if players:
                # Enrolment uses the local ear only; start the microphone without the agent.
                audio.muted = True
                audio.start(lambda _b: None)
                enrol(robot, ear, players)
                audio.stop()
                audio.muted = False
            if diarizer is not None and diarizer.wait_connected(2.0):
                log.info("live diarizer connected")
            if HEAD_SWAY and not args.no_sway and not robot.simulated:
                sway = HeadSway(robot._mini, lambda: audio.level_db).start()
            conversation.start_session()
            stop_watch = threading.Event()
            threading.Thread(
                target=table.watch_for_barge_in, args=(transcriber, stop_watch), name="barge-in", daemon=True
            ).start()
            log.info("talking (Ctrl+C to stop); players: %s", ", ".join(registry.names) or "unknown voices")
            diary.write("session_start", agent_id=agent_id, players=registry.names)
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        if sway is not None:
            sway.stop()
        try:
            stop_watch.set()
        except NameError:
            pass
        if conversation is not None:
            _end_session(conversation)
        if diarizer is not None:
            diarizer.stop()
        diary.write("session_end")
    return 0


def _wait_connected(conversation: Any, timeout: float) -> bool:
    """``start_session`` connects on a thread; wait until the websocket is up before sending anything."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(conversation, "_ws", None) is not None:
            time.sleep(0.3)  # let the initiation metadata land
            return True
        time.sleep(0.1)
    return False


def _end_session(conversation: Any, timeout: float = 5.0) -> None:
    """Close the agent session without hanging on a broken socket."""

    def close() -> None:
        try:
            conversation.end_session()
            conversation.wait_for_session_end()
        except Exception as exc:
            log.warning("ending the session: %s", exc)

    worker = threading.Thread(target=close, name="end-session", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        log.warning("the agent session did not close in %.0fs; leaving it", timeout)


def _terminate(*_: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    sys.exit(main())
