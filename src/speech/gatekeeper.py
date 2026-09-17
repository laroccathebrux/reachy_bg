"""Local addressee gate: hold each utterance, transcribe it here, decide, then release or discard.

    keeper = Gatekeeper(audio, transcriber, TurnLogger(), humans=2, names=registry.names,
                        my_investigator=lambda: game.robot_investigator.name, ...)
    ...                                     # the local ear closes an utterance:
    verdict = keeper.judge(utterance, speaker_name, score)     # audio released to the agent or dropped

The ElevenLabs agent only ever hears the utterances the rules say are for the robot, so
"stay quiet" costs no cloud turn and no tokens. With ``answer_everything`` the deciding stops
and every utterance is released - the rules still run and are still logged, so the sessions stay
comparable, and the robot's own echo is still dropped, because answering itself is not answering
the table. The decision is made on the local Whisper
transcript once the VAD closes the utterance; :class:`NameSpotter` runs earlier, on the voice
in progress, and opens the gate as soon as the robot's name is heard so that the rest of the
sentence streams live (no added latency for the most common way of addressing the robot).

Every judged utterance is one line of ``data/game_logs/addressee.jsonl`` with ``shadow=False``
when the gate controlled the audio and ``shadow=True`` when it only watched (``--always-answer``
or a solo table, where everything passes).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.logger import get_logger
from src.speech.addressee import Decision, TurnLogger, decide, looks_like_echo, mentions_robot

log = get_logger(__name__)

SWITCH_MIN_WORDS = 3  # shorter fragments get a wrong language too easily to restart the session
SWITCH_MIN_CONFIDENCE = 0.8
ECHO_SLACK_S = 0.5  # an utterance that started this long after playback ended can still be echo
MIN_OVER_S = 0.7  # voice left after the robot stopped, below which there is nothing to judge


@dataclass(frozen=True)
class Verdict:
    """What the gate did with one utterance."""

    route: str  # released | early | discarded | switched | echo_gate | no_speech | passed,
    # or what the robot handled itself: my_turn | setup
    decision: Decision
    text: str
    language: str
    forwarded: int  # 250 ms frames sent to the agent
    dropped: int  # 250 ms frames the agent never heard
    decision_ms: int  # from the VAD closing the utterance to the audio being released or dropped
    whisper_ms: int


class Gatekeeper:
    def __init__(
        self,
        audio: Any,
        transcriber: Any,
        turn_log: TurnLogger,
        *,
        active: bool = True,
        answer_everything: bool = False,
        humans: int | None = None,
        names: tuple[str, ...] = (),
        spoken_recently: Callable[[], str] = lambda: "",
        robot_spoke_at: Callable[[], float | None] = lambda: None,
        voice_language: Callable[[], str] = lambda: "",
        on_switch: Callable[[str, str, str], None] | None = None,
        my_investigator: Callable[[], str] = lambda: "",
        on_addressed: Callable[..., str | None] | None = None,
        wants_setup: Callable[[str], bool] = lambda text: False,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.audio = audio
        self.transcriber = transcriber
        self.turn_log = turn_log
        self.active = active
        # "Answer everything": the rules stop deciding and every utterance goes to the agent.
        # The audio is still held and still transcribed, because that is what the robot's own
        # ear needs - the self-echo check so it does not answer itself, the language switch, and
        # the turn call and the briefing it handles without the cloud. Without the holding, the
        # agent would already have heard a turn call by the time the robot decided to take it,
        # and both of them would answer.
        self.answer_everything = answer_everything
        if answer_everything:
            log.info("gatekeeper: the gate is off - everything is answered, nothing is dropped")
        self.humans = humans
        self.names = tuple(names)
        self.spoken_recently = spoken_recently
        self.robot_spoke_at = robot_spoke_at
        self.voice_language = voice_language
        self.on_switch = on_switch
        # Read at decision time, not at construction: the robot is given its investigator
        # during the spoken setup, after the ear is already listening.
        self.my_investigator = my_investigator
        # What the robot answers by itself, before the agent is given anything: taking its own
        # turn, and writing down a spoken setup (src/integration/game_session.py). It is given
        # the transcript, the language, the rule that addressed it and the speaker, and returns
        # the route it handled the utterance as - or None, and the agent hears it as usual.
        self.on_addressed = on_addressed
        # "Is this the briefing I am still waiting for?" - answered by the game session, which
        # is the only thing that knows what the robot still has to be told.
        self.wants_setup = wants_setup
        self.clock = clock
        self.asr_lock = threading.Lock()  # one Whisper call at a time (the spotter shares it)
        self.last_addressed = ""
        # Who last spoke to the robot and when: while that voice holds the floor, its next
        # sentences are for the robot too (see addressee.decide's seconds_since_addressed).
        self.addressed_at: float | None = None
        self.addressed_label = ""
        self.early_release_at: float | None = None
        self.early_text = ""
        self.early_language = ""
        self.verdicts: list[Verdict] = []

    def _floor_seconds(self, speaker: str, label: str) -> float | None:
        """How long ago this same voice last spoke to the robot, or None if it was not this one.

        The voiceprint name is the better key; when nobody is enrolled the diarizer's anonymous
        label still separates one speaker from another. With neither, the floor is not opened:
        a table where no voice can be told apart is exactly where "everything continues" would
        send the other player's sentences to the agent.
        """
        if self.addressed_at is None:
            return None
        if speaker or self.last_addressed:
            same = bool(speaker) and speaker == self.last_addressed
        elif label or self.addressed_label:
            same = bool(label) and label == self.addressed_label
        else:
            same = False
        return self.clock() - self.addressed_at if same else None

    def _floor_opened(self, speaker: str, label: str) -> None:
        self.last_addressed = speaker
        self.addressed_label = label
        self.addressed_at = self.clock()

    # ------------------------------------------------------------------ early release
    def early_release(self, text: str, language: str) -> int:
        """The robot's name was heard while the voice is still running: let the rest stream live."""
        self.early_release_at = self.clock()
        self.early_text = text
        self.early_language = language
        return self.audio.pass_through() if self.active else 0

    # ------------------------------------------------------------------ judgement
    def _after_playback(self, utterance: Any) -> tuple[Any, float | None]:
        """``(audio, echo_until)``: the part of the voice the robot was not making.

        ``echo_until`` is the monotonic time the robot's own voice stopped reaching the
        microphone, or None when it was not speaking during this one. Frames that arrived
        before it are its echo and neither transcribed nor forwarded.
        """
        cut = self.audio.last_played_at + self.audio.gate_tail_s
        audio = utterance.audio
        rate = utterance.sample_rate or 1
        if utterance.started_at > cut or utterance.ended_at - cut < MIN_OVER_S:
            return audio, None
        keep = int((utterance.ended_at - cut) * rate)
        if keep <= 0 or keep >= audio.size:
            return audio, None
        log.info(
            "voice over the robot: dropping %.1fs of echo, judging the last %.1fs",
            (audio.size - keep) / rate,
            keep / rate,
        )
        return audio[-keep:], cut

    def judge(self, utterance: Any, speaker: str = "", score: float = 0.0, label: str = "") -> Verdict:
        until = utterance.ended_at
        early_at = self.early_release_at
        if early_at is not None and utterance.started_at <= early_at <= until + 1.0:
            self.early_release_at = None
            verdict = Verdict(
                "early", Decision(True, "name", 0.95), self.early_text, self.early_language, 0, 0, 0, 0
            )
            self._floor_opened(speaker, label)
            return self._done(verdict, utterance, speaker, score, label, None)

        # The whole voice fell inside the robot's own playback and no barge-in released it: the
        # echo gate already swallowed those frames; nothing to transcribe.
        if utterance.speech_ended_at <= self.audio.last_played_at + self.audio.gate_tail_s:
            dropped = self.audio.discard_utterance(until) if self.active else 0
            verdict = Verdict("echo_gate", Decision(False, "self_echo", 0.9), "", "", 0, dropped, 0, 0)
            return self._done(verdict, utterance, speaker, score, label, None)

        # A player who starts talking over the end of the robot's sentence leaves the local VAD
        # one unbroken voice whose first half is the robot's own echo. Judging that whole thing
        # asks the wrong question: most of the words are the robot's, looks_like_echo says echo,
        # and self_echo is the one verdict that survives even with the gate off - so the player's
        # sentence went in the bin with the echo. On 2026-09-17 "Dá uma olhada, porque eu já te
        # falei isso" was dropped that way. Only what was said after the speaker fell silent is
        # theirs, so only that is transcribed, and only that is forwarded.
        audio, echo_until = self._after_playback(utterance)

        started = self.clock()
        with self.asr_lock:
            result = self.transcriber.transcribe(
                audio, utterance.sample_rate, fallback_language=self.voice_language(), fast=True
            )
        whisper_ms = int((self.clock() - started) * 1000)
        text = (result.text or "").strip()
        language = result.language or self.voice_language()
        if not text:
            dropped = self.audio.discard_utterance(until) if self.active else 0
            verdict = Verdict(
                "no_speech", Decision(False, "no_speech", 0.5), "", language, 0, dropped, 0, whisper_ms
            )
            return self._done(verdict, utterance, speaker, score, label, None)

        spoke_at = self.robot_spoke_at()
        since = None if spoke_at is None else round(max(0.0, utterance.started_at - spoke_at), 1)
        # A voiceprint that matched a player settles it: the robot's own voice scores far below
        # the match threshold against every enrolled player (measured over a session: echo at most
        # 0.25, a player 0.45 and up), so a name here means a person spoke, whatever the words
        # look like. Without this the text alone decided, and a sentence that repeated the robot's
        # own words back at it - which is what people do - read as echo.
        may_be_echo = not speaker and utterance.started_at <= self.audio.last_played_at + ECHO_SLACK_S
        if may_be_echo and looks_like_echo(text, self.spoken_recently()):
            decision = Decision(False, "self_echo", 0.9)
        else:
            decision = decide(
                text,
                language,
                seconds_since_robot_spoke=since,
                humans_present=self.humans,
                follow_up_ok=bool(speaker) and speaker == self.last_addressed,
                other_names=[n for n in self.names if n != speaker],
                my_investigator=self.my_investigator(),
                seconds_since_addressed=self._floor_seconds(speaker, label),
                setup_hint=self.wants_setup(text),
            )

        # With the gate off the robot's own ear looks at everything it hears: it is what keeps
        # its memory of the game current when the rules would have stayed quiet. It still only
        # *keeps* an utterance when it did something with it; otherwise the agent gets it.
        if (
            self.on_addressed is not None
            and decision.reason != "self_echo"
            and (decision.addressed or self.answer_everything)
        ):
            handled = None
            try:
                handled = self.on_addressed(text, language, decision.reason, speaker)
            except Exception as exc:  # handling it must not swallow the utterance silently
                log.exception("gatekeeper: handling %r failed (%s)", decision.reason, exc)
            if handled:
                dropped = self.audio.discard_utterance(until) if self.active else 0
                self._floor_opened(speaker, label)
                verdict = Verdict(handled, decision, text, language, 0, dropped, 0, whisper_ms)
                return self._done(verdict, utterance, speaker, score, label, since)

        # With the gate off the rules still run and are still logged - that file is the dataset
        # the learned classifier will be trained on, and a session with the gate off has to stay
        # comparable with one with it on. Only the routing changes: everything is released. The
        # robot's own echo is the exception, because answering itself is not answering the table.
        #
        # The text check alone is not enough for that exception. A short fragment of the robot's
        # own sentence - "Uma coisa?" out of "...mais alguma coisa?" - carries too few content
        # words to be recognised, and on 2026-09-17 one was forwarded and the agent stopped
        # answering the table for the rest of the session. The voiceprint is the better witness:
        # it matched nobody at all, and the voice began while the speaker was playing. That is
        # the robot, not a player. Only meaningful when somebody is enrolled - with an empty
        # registry no voice ever has a name and this would drop the whole table.
        anonymous_echo = (
            bool(self.names)
            and not speaker
            and utterance.started_at <= self.audio.last_played_at + ECHO_SLACK_S
        )
        if anonymous_echo and not decision.addressed:
            log.info("no voice matched and it began under the robot's own: %r", text[:60])
        forced = (
            self.answer_everything
            and not decision.addressed
            and decision.reason != "self_echo"
            and not anonymous_echo
        )

        forwarded = dropped = 0
        confident_switch = (
            self.on_switch is not None
            and language != self.voice_language()
            and result.language_confidence >= SWITCH_MIN_CONFIDENCE
            and len(text.split()) >= SWITCH_MIN_WORDS
        )
        if not self.active:
            route = "passed"
            if decision.addressed and confident_switch:
                route = "switched"
        elif forced:
            forwarded = self.audio.release_utterance(until, since=echo_until)
            route = "released"
        elif decision.addressed and confident_switch:
            dropped = self.audio.discard_utterance(until)
            route = "switched"
        elif decision.addressed:
            forwarded = self.audio.release_utterance(until, since=echo_until)
            route = "released"
        else:
            dropped = self.audio.discard_utterance(until)
            route = "discarded"
        if decision.addressed:
            self._floor_opened(speaker, label)
        verdict = Verdict(route, decision, text, language, forwarded, dropped, 0, whisper_ms)
        verdict = self._done(verdict, utterance, speaker, score, label, since)
        if route == "switched" and self.on_switch is not None:
            self.on_switch(language, text, f"whisper {result.language_confidence:.2f}")
        return verdict

    def _done(
        self, verdict: Verdict, utterance: Any, speaker: str, score: float, label: str, since: float | None
    ) -> Verdict:
        verdict = Verdict(
            verdict.route,
            verdict.decision,
            verdict.text,
            verdict.language,
            verdict.forwarded,
            verdict.dropped,
            int((self.clock() - utterance.ended_at) * 1000),
            verdict.whisper_ms,
        )
        self.verdicts = (self.verdicts + [verdict])[-50:]
        self.turn_log.log(
            verdict.decision,
            text=verdict.text,
            language=verdict.language,
            speaker=speaker,
            speaker_score=round(score, 3),
            diart_label=label,
            seconds_since_robot_spoke=since,
            shadow=not self.active,
            gate_off=self.answer_everything,
            route=verdict.route,
            forwarded=verdict.forwarded,
            dropped=verdict.dropped,
            decision_ms=verdict.decision_ms,
            whisper_ms=verdict.whisper_ms,
            speech_s=round(utterance.speech_s, 2),
            capture=str(utterance.path) if getattr(utterance, "path", None) else "",
        )
        return verdict


class NameSpotter:
    """Transcribe the voice in progress and report when the robot's name is heard.

    Same cadence as the barge-in check: nothing before ``min_ms`` of voice, then again every
    ``recheck_ms`` of new voice. Whisper is skipped when the gatekeeper is already using it.
    """

    def __init__(
        self,
        mic: Any,
        keeper: Gatekeeper,
        *,
        min_ms: int = 900,
        recheck_ms: int = 1200,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.mic = mic
        self.keeper = keeper
        self.min_ms = min_ms
        self.recheck_ms = recheck_ms
        self.clock = clock
        self._checked_ms = 0
        self.checks = 0

    def __call__(self) -> str | None:
        voice_ms = int(self.mic.voice_ms)
        if voice_ms < self.min_ms:
            if voice_ms == 0:
                self._checked_ms = 0
            return None
        if self._checked_ms and voice_ms - self._checked_ms < self.recheck_ms:
            return None
        if not self.keeper.asr_lock.acquire(blocking=False):
            return None
        try:
            audio = self.mic.current_audio()
            if audio.size == 0:
                return None
            self._checked_ms = voice_ms
            self.checks += 1
            started = self.clock()
            result = self.keeper.transcriber.transcribe(
                audio, fallback_language=self.keeper.voice_language(), fast=True
            )
        finally:
            self.keeper.asr_lock.release()
        text = (result.text or "").strip()
        heard = bool(text) and mentions_robot(text)
        log.info(
            "name check %d (%.1fs voice, %.2fs): %r -> %s",
            self.checks,
            voice_ms / 1000,
            self.clock() - started,
            text,
            "named" if heard else "no name",
        )
        if heard:
            self.keeper.early_release(text, result.language or self.keeper.voice_language())
            return text
        return None


__all__ = ["Gatekeeper", "NameSpotter", "Verdict", "SWITCH_MIN_WORDS", "SWITCH_MIN_CONFIDENCE"]
