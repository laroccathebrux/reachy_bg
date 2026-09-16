"""Hearing "é a vez da Lily Chen" and answering with a move, without spending a cloud turn.

    taker = TurnTaker(game, say=speak, language="pt-BR")
    taker.handle("é a vez da Lily Chen", "pt-BR")   # -> True: it took the turn itself

The addressee rules already know that a turn call naming the robot's own investigator is for the
robot (``reason="my_turn"``, the same confidence as calling its name). This is what happens next:
the audio is dropped instead of being released to the ElevenLabs agent, the decision is made
locally by ``src.strategy.decide``, and the robot says the reason that came out of it, word for
word, in the voice of the table's language.

**The agent never sees a turn call.** Not to save the minutes, though it does: the sentence the
decision produced is already a finished spoken answer, in the right language, with the game terms
in English, and handing it to a conversational model to "say" is handing it a chance to rewrite
it into something the robot did not decide. What is said is what was chosen.

The robot announces; the person moves the piece. Nothing here touches the board - the camera
sees the move happen, the way it does for every other piece.

Only two lines are fixed text rather than the model's words: "I cannot act this turn" and "I do
not know which investigator I am playing", because there is no decision to explain and no reason
to pay for a sentence saying so.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from src.logger import get_logger
from src.speech.addressee import is_turn_call, names_investigator
from src.strategy.decide import Decision, decide
from src.strategy.game import GameState

log = get_logger(__name__)

# Fixed lines, one per spoken language: there is no decision behind them to explain.
CANNOT_ACT = {
    "pt-BR": "Não posso agir neste turno.",
    "en-US": "I cannot take an action this turn.",
}
NO_INVESTIGATOR = {
    "pt-BR": "Ainda não sei qual investigador eu jogo.",
    "en-US": "I do not know which investigator I am playing yet.",
}


class TurnTaker:
    """Takes the robot's own turn out loud when the table hands it over.

    ``say(text, language)`` is how the sentence reaches the speaker; in ``talk.py`` it is the
    local TTS written to the same output stream the agent uses, so the echo gate and the head
    sway work exactly as they do for the agent's own voice.
    """

    def __init__(
        self,
        game: GameState,
        *,
        say: Callable[[str, str], None],
        language: str = "pt-BR",
        plan: Any = None,
        decide_fn: Callable[..., Decision] = decide,
        on_decision: Callable[[Decision, str], None] | None = None,
        background: bool = True,
    ) -> None:
        self.game = game
        self.say = say
        self.language = language
        self.plan = plan
        self.decide_fn = decide_fn
        self.on_decision = on_decision
        self.background = background
        self.last: Decision | None = None
        self.turns = 0
        self._busy = threading.Lock()

    @property
    def investigator(self) -> str:
        state = self.game.robot_investigator
        return state.name if state is not None else ""

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def is_for_me(self, text: str, language: str) -> bool:
        """Is this the table handing the turn to the investigator the robot plays?"""
        mine = self.investigator
        return bool(mine) and names_investigator(text, mine) and is_turn_call(text, language)

    def handle(self, text: str, language: str = "") -> bool:
        """Take the turn if this sentence hands it over. True means the robot dealt with it.

        A second turn call while one is being decided is swallowed rather than queued: the table
        repeating itself must not produce two moves.
        """
        language = language or self.language
        if not self.is_for_me(text, language):
            return False
        if self.busy:
            log.info("turn: already deciding; ignoring %r", text)
            return True
        if self.background:
            threading.Thread(target=self._take, args=(language,), name="turn-taker", daemon=True).start()
        else:
            self._take(language)
        return True

    def decide_now(self, language: str = "") -> Decision | None:
        """Work the turn out and return it, without saying anything.

        This is what the agent's ``take_turn`` tool calls: with one voice at the table, the
        robot decides and the agent speaks, so the decision has to come back as text rather
        than go to a speaker of its own.
        """
        language = language or self.language
        if self.game.robot_investigator is None:
            return None
        with self._busy:
            started = time.monotonic()
            try:
                decision = self.decide_fn(self.game, language=language, plan=self.plan)
            except Exception as exc:  # a decision that crashes must not take the turn down
                log.exception("turn: the decision failed (%s)", exc)
                return None
            self.last = decision
            self.turns += 1
            log.info(
                "turn %d: %s (%s, %.1fs)",
                self.turns,
                decision.plan.describe() if decision.plan else "no action",
                decision.chosen_by,
                time.monotonic() - started,
            )
            if self.on_decision is not None:
                self.on_decision(decision, self.sentence(decision, language))
            return decision

    def _take(self, language: str) -> None:
        with self._busy:
            started = time.monotonic()
            if self.game.robot_investigator is None:
                self.say(NO_INVESTIGATOR.get(language, NO_INVESTIGATOR["en-US"]), language)
                return
            try:
                decision = self.decide_fn(self.game, language=language, plan=self.plan)
            except Exception as exc:  # a decision that crashes must not take the conversation down
                log.exception("turn: the decision failed (%s)", exc)
                return
            self.last = decision
            self.turns += 1
            text = self.sentence(decision, language)
            log.info(
                "turn %d: %s (%s, %.1fs total)",
                self.turns,
                decision.plan.describe() if decision.plan else "no action",
                decision.chosen_by,
                time.monotonic() - started,
            )
            if self.on_decision is not None:
                self.on_decision(decision, text)
            self.say(text, language)

    @staticmethod
    def sentence(decision: Decision, language: str) -> str:
        """What the robot says: the reason as decided, plus the one question it would ask.

        The reason is never rewritten. ``decision.ask`` is added when there is one, because it is
        the thing that would change the move - "quanto vale o Bull Whip?" - and the table can
        answer before the piece is moved. Only that field is spoken: the other entries in
        ``questions`` are the generator's own notes, written in English for the log, and a
        Portuguese table should not be read an English sentence.
        """
        if decision.plan is None:
            return CANNOT_ACT.get(language, CANNOT_ACT["en-US"])
        return " ".join(p for p in (decision.reason.strip(), decision.ask.strip()) if p)


__all__ = ["CANNOT_ACT", "NO_INVESTIGATOR", "TurnTaker"]
