"""Table conversation on an ElevenLabs agent, with the robot's ears, memory and body kept local.

    uv run python -m src.integration.talk                     # daemon on :8000, Mac mic, robot speaker
    uv run python -m src.integration.talk --no-robot          # gestures logged, robot speaker still used
    uv run python -m src.integration.talk --players "Ana,Bruno"   # enrol voices first (local)
    uv run python -m src.integration.talk --device "MacBook Pro Microphone"
    uv run python -m src.integration.talk --humans 2          # two people at the table: the gate is on
    uv run python -m src.integration.talk --always-answer     # no gate: the agent hears everything
    uv run python -m src.integration.talk --new-game          # forget the saved game and set up again
    uv run python -m src.integration.talk --demo-game --game-plan   # the robot plays its own investigator

The agent does ASR, LLM, TTS and turn-taking in the cloud (about half a second per turn)
and calls back into this process for rules and knowledge (Qdrant). Locally, the same
microphone audio feeds the voice activity detector, the voiceprints and the live diarizer,
so every heard utterance is attributed to a player and judged by the addressee rules *before*
the agent hears it: the audio of an utterance is held back, transcribed by the local Whisper,
and only released to the agent when the rules say it was for the robot (see
src/speech/gatekeeper.py). Table talk that is not for the robot never reaches the cloud, so it
costs no turn and no tokens. With ``--always-answer`` (or one person at the table) the gate
only watches and its decisions are logged as shadow decisions. Every decision goes to
data/game_logs/addressee.jsonl; everything heard and said to data/game_logs/conversation.jsonl.

The robot also plays, and remembers the game it is playing between runs
(src/integration/game_session.py). The setup is heard rather than typed: while it still does not
know the Ancient One or which investigator is its own, a sentence that is not a question goes to
the local extractor instead of to the agent, the robot says back what it wrote down and asks for
the next missing thing, and every change is written to ``--game`` (data/game_logs/game_state.json
by default) at once. Starting again reads the file back and the robot says what it remembers, so
a stale game announces itself; ``--new-game`` forgets it and ``--no-game`` is conversation only.

When the gate hears its own investigator's turn being called, the audio is dropped instead of
being released, the move is decided locally by src/strategy/decide.py and the robot says the
reason out of its own TTS. The agent is never asked to say a move it did not choose, and neither
the setup nor a turn costs any agent minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_LANGUAGE,
    DIARIZER_URL,
    ELEVENLABS_API_KEY,
    GAME_LOG_DIR,
    HEAD_SWAY,
    validate_config,
)
from src.integration.game_session import GameSession, looks_like_setup
from src.logger import get_logger
from src.robot.reachy import AGREE_MOVES, GREETING_MOVES, Robot
from src.robot.sway import HeadSway
from src.speech.addressee import TurnLogger
from src.speech.agent_audio import RobotAudioInterface
from src.speech.asr import Transcriber
from src.speech.barge_in import EchoAwareBargeIn
from src.speech.diarization import DiarizerClient
from src.speech.eleven_agent import client_tools, ensure_agent, session_override
from src.speech.gatekeeper import Gatekeeper, NameSpotter
from src.speech.language import detect_language
from src.speech.microphone import Segmenter, Utterance, write_wav
from src.speech.speakers import MIN_ENROLL_S, SpeakerRegistry
from src.speech.tts import TTSError, synthesize
from src.strategy.game import GameState

log = get_logger(__name__)

ENROL_PROMPTS = {
    "pt-BR": "{name}, diga uma frase longa, de uns cinco segundos, para eu aprender a sua voz.",
    "en-US": "{name}, say a long sentence, about five seconds, so I can learn your voice.",
}
ENROL_THANKS = {"pt-BR": "Obrigado, {name}.", "en-US": "Thank you, {name}."}
SIDECAR_DIR = Path(__file__).resolve().parents[2] / "tools" / "live-diarizer"


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
    on_utterance: Callable[[Utterance, str, float], None] | None = None  # the gate, once wired

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
        if self.on_utterance is not None:
            try:
                self.on_utterance(utterance, name, score)
            except Exception as exc:
                log.warning("utterance callback failed: %s", exc)

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
        self.keeper: Gatekeeper | None = None  # set by main once the table size is known
        self._judge_queue: queue.Queue[tuple[Utterance, str, float] | None] = queue.Queue()
        self._judge: threading.Thread | None = None

    # ------------------------------------------------------------------ the local gate
    def on_utterance(self, utterance: Utterance, name: str, score: float) -> None:
        """The local VAD closed an utterance (audio thread): judge it on the gate thread, in order."""
        self.audio.utterance_ended()
        if self.keeper is None:
            return
        if self._judge is None:
            self._judge = threading.Thread(target=self._judge_loop, name="gatekeeper", daemon=True)
            self._judge.start()
        self._judge_queue.put((utterance, name, score))

    def _judge_loop(self) -> None:
        while True:
            item = self._judge_queue.get()
            if item is None:
                return
            utterance, name, score = item
            label = ""
            if self.ear.diarizer is not None and self.ear.diarizer.connected:
                label, _ = self.ear.diarizer.speaker_between(
                    wall_time(utterance.started_at), wall_time(utterance.speech_ended_at)
                )
            try:
                verdict = self.keeper.judge(utterance, name, score, label)  # type: ignore[union-attr]
            except Exception as exc:
                log.warning("gate decision failed (%s); releasing the utterance", exc)
                self.audio.release_utterance(utterance.ended_at)
                continue
            self.diary.write(
                "gate",
                route=verdict.route,
                text=verdict.text,
                language=verdict.language,
                speaker=name,
                score=round(score, 3),
                diart=label,
                addressed=verdict.decision.addressed,
                reason=verdict.decision.reason,
                forwarded=verdict.forwarded,
                dropped=verdict.dropped,
                decision_ms=verdict.decision_ms,
                whisper_ms=verdict.whisper_ms,
            )
            if verdict.route == "echo_gate":
                continue
            log.info(
                "gate [%s %.2f%s] %s: %r  (%s, %s; whisper %d ms, decided %d ms after the voice ended)",
                name or "?",
                score,
                f" {label}" if label else "",
                verdict.route,
                verdict.text,
                "for me" if verdict.decision.addressed else "not for me",
                verdict.decision.reason,
                verdict.whisper_ms,
                verdict.decision_ms,
            )

    def on_user_transcript(self, text: str) -> None:
        """What the agent's ASR made of the audio the gate released (the diary keeps both texts)."""
        heard = self.ear.last_utterance()
        name, score = ("", 0.0) if heard is None else (heard[1], heard[2])
        language = detect_language(text)
        self.diary.write("heard", text=text, speaker=name, score=round(score, 3))
        self.last_addressed = name
        # The gate already checked the language with Whisper before releasing; this is the
        # fallback for the cases where Whisper was unsure and the agent's transcript is not.
        if language != self.voice_language and len(text.split()) >= 3:
            self.switch_language(language, text, "transcript")
        log.info("heard [%s %.2f]: %s", name or "?", score, text)

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

    def watch_voice(self, transcriber: Transcriber, stop: threading.Event) -> None:
        """Whisper on the voice in progress: a player's words over the robot release the echo gate;
        the robot's name while it listens opens the addressee gate before the sentence ends."""
        checker: EchoAwareBargeIn | None = None
        spotter = (
            NameSpotter(self.ear, self.keeper) if self.keeper is not None and self.keeper.active else None
        )
        while not stop.is_set():
            time.sleep(0.1)
            if not self.audio.gated:
                checker = None
                if spotter is not None and self.audio.holding:
                    heard = spotter()
                    if heard:
                        self.diary.write("early_release", heard=heard)
                        log.info("named while listening (%r); the rest streams live", heard)
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


def speak_pcm(audio: Any, text: str, language: str, *, table: Any = None) -> None:
    """Say a line of our own through the agent's own speaker stream.

    The robot has two voices in this process: the agent's, which arrives as PCM over the
    WebSocket, and this one, synthesized locally for what the robot decided by itself. Both are
    written to the same output queue, so the echo gate, the barge-in check and the head sway see
    them as the same voice - which they are, to everybody at the table.
    """
    log.info("robot (local): %s", text)
    try:
        clip = synthesize(text, language)
    except TTSError as exc:
        log.warning("turn: could not synthesize (%s)", exc)
        return
    with wave.open(str(clip.path), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    if table is not None:
        table.on_agent_response(text)  # the barge-in and echo checks compare against what is said
    audio.output(pcm)


def build_game_session(args: Any, audio: Any, table: Any, diary: Diary) -> GameSession | None:
    """The game this session is playing: read back from its file, or started from nothing.

    The file is the memory. It is written every time the robot learns something, so closing the
    conversation loses nothing, and reopening makes the robot say what it remembers - which is
    also how a stale game gets caught: it tells the table it is still playing yesterday's, and
    the table says otherwise.
    """
    if args.no_game:
        return None
    path = Path(args.game)

    def say(text: str, language: str) -> None:
        speak_pcm(audio, text, language, table=table)

    if args.demo_game:
        from src.strategy.turn import demo_game

        session = GameSession(demo_game(), say=say, path=path, language=DEFAULT_LANGUAGE)
    elif args.new_game:
        session = GameSession(GameState(), say=say, path=path, language=DEFAULT_LANGUAGE)
    else:
        session = GameSession.open(path, say=say, language=DEFAULT_LANGUAGE)
    if args.game_plan and session.game.ready:
        from src.strategy import plan as planner

        plan = planner.make_plan(
            session.game, language=DEFAULT_LANGUAGE, advice=planner.advice_for(session.game)
        )
        session.taker.plan = plan
        log.info("plan for this game:\n%s", plan.as_text())
        diary.write("game_plan", **plan.record())
    session.taker.on_decision = lambda decision, text: diary.write("turn", said=text, **decision.record())
    session.on_setup = lambda reading, text: diary.write("setup", heard=text, **reading.record())
    log.info("game: %s", session.game.briefing())
    diary.write("game_opened", briefing=session.game.briefing(), missing=session.game.missing())
    return session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-robot", action="store_true", help="no daemon: gestures logged")
    parser.add_argument("--device", default=None, help="input device name substring or index")
    parser.add_argument("--output-device", default=None, help="output device name substring")
    parser.add_argument("--players", default="", help="comma-separated names to enrol by voice at the start")
    parser.add_argument("--no-diarizer", action="store_true", help="neither start nor use the diart sidecar")
    parser.add_argument("--no-sway", action="store_true", help="do not move the head with the voice")
    parser.add_argument(
        "--humans",
        type=int,
        default=None,
        help="people at the table (default: enrolled voices); 1 = solo, everything is for the robot",
    )
    parser.add_argument(
        "--always-answer",
        action="store_true",
        help="no addressee gate: the agent hears everything (decisions logged as shadow)",
    )
    parser.add_argument(
        "--game",
        default=str(Path(GAME_LOG_DIR) / "game_state.json"),
        help="the file the game is remembered in; read at the start, written at every change",
    )
    parser.add_argument("--new-game", action="store_true", help="forget the saved game and start again")
    parser.add_argument("--no-game", action="store_true", help="conversation only: the robot does not play")
    parser.add_argument("--demo-game", action="store_true", help="the demo setup of src.strategy.turn")
    parser.add_argument(
        "--game-plan",
        action="store_true",
        help="write the plan for the game after loading it, and follow it on every turn",
    )
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
    log.info("whisper warm-up (gate and barge-in checks): %.1fs", transcriber.warm_up())
    input_device = args.device if args.device is not None else AUDIO_INPUT_DEVICE
    sidecar = None if args.no_diarizer else start_sidecar(input_device)
    diarizer = None if args.no_diarizer else DiarizerClient().start()
    ear = LocalEar(Segmenter(), registry, diarizer, AUDIO_CAPTURE_DIR if SAVE_CAPTURES else None)
    audio = RobotAudioInterface(
        input_device, args.output_device if args.output_device is not None else AUDIO_OUTPUT_DEVICE
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
                alive = threading.Event()  # cleared before the session is closed: its tools stop searching
                alive.set()
                conv = TableConversation(
                    client,
                    agent_id,
                    requires_auth=True,
                    audio_interface=audio,
                    client_tools=client_tools(
                        on_call=table.on_tool, active=alive.is_set
                    ),  # fresh per session
                    config=ConversationInitiationData(
                        conversation_config_override=session_override(language)
                    ),
                    callback_agent_response=table.on_agent_response,
                    callback_agent_response_correction=table.on_agent_correction,
                    callback_user_transcript=table.on_user_transcript,
                    callback_latency_measurement=table.on_latency,
                    on_raw=table.on_raw_event,
                )
                conv.alive = alive
                table.conversation = conv
                table.voice_language = language
                return conv

            def restart(language: str, text: str) -> None:
                nonlocal conversation
                conversation.alive.clear()
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
            humans = args.humans if args.humans is not None else (len(registry.names) or None)
            gate_on = not args.always_answer and humans != 1
            session = build_game_session(args, audio, table, diary)
            table.keeper = Gatekeeper(
                audio,
                transcriber,
                table.turn_log,
                active=gate_on,
                humans=humans,
                names=tuple(registry.names),
                spoken_recently=lambda: table.spoken_recently,
                robot_spoke_at=lambda: table.robot_spoke_at,
                voice_language=lambda: table.voice_language,
                on_switch=table.switch_language,
                my_investigator=(lambda: session.taker.investigator) if session is not None else (lambda: ""),
                on_addressed=session.handle if session is not None else None,
                wants_setup=(
                    (lambda text: session.wants_setup and looks_like_setup(text))
                    if session is not None
                    else (lambda text: False)
                ),
            )
            audio.hold_utterances = gate_on
            ear.on_utterance = table.on_utterance
            log.info(
                "addressee gate %s (%s at the table%s)",
                "on: the agent only hears what is for the robot"
                if gate_on
                else "off: the agent hears everything",
                "?" if humans is None else humans,
                ", --always-answer" if args.always_answer else "",
            )
            if diarizer is not None and diarizer.wait_connected(2.0):
                log.info("live diarizer connected")
            if HEAD_SWAY and not args.no_sway and not robot.simulated:
                sway = HeadSway(robot._mini, lambda: audio.level_db).start()
            conversation.start_session()
            stop_watch = threading.Event()
            threading.Thread(
                target=table.watch_voice, args=(transcriber, stop_watch), name="voice-watch", daemon=True
            ).start()
            log.info("talking (Ctrl+C to stop); players: %s", ", ".join(registry.names) or "unknown voices")
            if session is not None:
                greeting = session.greeting()
                if greeting:
                    speak_pcm(audio, greeting, DEFAULT_LANGUAGE, table=table)
            diary.write(
                "session_start", agent_id=agent_id, players=registry.names, humans=humans, gate=gate_on
            )
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
        stop_sidecar(sidecar)
        try:
            if session is not None:
                session.save()
        except NameError:
            pass
        diary.write("session_end")
    return 0


def sidecar_port_open(url: str = DIARIZER_URL, timeout: float = 0.3) -> bool:
    host, _, port = url.split("//", 1)[-1].partition(":")
    try:
        with socket.create_connection((host or "127.0.0.1", int(port.split("/")[0] or 8765)), timeout):
            return True
    except OSError:
        return False


def start_sidecar(device: str, sidecar_dir: Path = SIDECAR_DIR) -> subprocess.Popen | None:
    """Launch ``tools/live-diarizer`` when it is installed and not already listening.

    Its output goes to ``data/game_logs/live_diarizer.log``; the client connects on its own
    once the models are loaded (about 20 s). Returns the process to stop at exit, or None.
    """
    if sidecar_port_open():
        log.info("live diarizer already listening at %s", DIARIZER_URL)
        return None
    if not (sidecar_dir / ".venv").exists():
        log.info("live diarizer not installed (%s); speaker labels off", sidecar_dir / ".venv")
        return None
    GAME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = (GAME_LOG_DIR / "live_diarizer.log").open("ab")
    command = ["uv", "run", "python", "-m", "live_diarizer"]
    if device:
        command += ["--device", device]
    # The sidecar has its own venv (numpy < 2); the variables of ours must not leak into it,
    # or its torch imports half of their packages from our site-packages and crash.
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    try:
        process = subprocess.Popen(
            command, cwd=sidecar_dir, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
    except OSError as exc:
        log.warning("could not start the live diarizer: %s", exc)
        return None
    log.info("live diarizer starting (pid %d); log: %s", process.pid, GAME_LOG_DIR / "live_diarizer.log")
    return process


def stop_sidecar(process: subprocess.Popen | None, timeout: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        process.kill()


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
