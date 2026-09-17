"""Thinking between rounds, with tools, while the humans play.

    thinking = Reflection(game).run()          # blocking, for scripts and tests
    thinking.intents                           # what the score will read next round
    thinking.refused                           # what the board would not support, and why

This is the one place in the project where the reasoning model is allowed to take its time. It
is **not** in the turn path and must never be put there: ``think=True`` was measured on this Mac
at 114 s against 4 s and rejected, because a table does not wait two minutes for a move. The
dead time while the humans take their turns is minutes long and costs nothing, so it is where a
slow loop belongs.

What the loop is for is the failure the measurements kept finding: **the facts are grounded and
the causal links are not.** Asked for a turn in one shot, the model picks first and then
justifies from whatever is in its context window - it explained a Prepare for Travel by the
Protective Amulet's +1 Will. A loop attacks exactly that, in two ways:

* it may **look things up before committing**, so a link can be checked rather than assumed, and
  what it looked up is carried on the intent it proposed;
* what it is allowed to produce is not prose but three shapes of intent whose targets the code
  checks against the board and the printed action list (``src.strategy.intent``). An invented
  link that names a space that does not exist is refused before it can become a move.

The loop is a plain cycle rather than the SDK's tool calling, because ``llm.ollama_client.chat``
has no tool protocol and a JSON step does the same work with nothing to go wrong in the
middle: each step the model answers either ``{"look_up": "..."}`` or ``{"intents": [...]}``,
and it is given what it asked for and asked again. It is stopped by a step budget, by a clock,
and by ``cancel()`` - which is what the round starting calls.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.llm.ollama_client import LLMError, chat
from src.llm.prompts import language_name
from src.logger import get_logger
from src.strategy import intent as intents_module
from src.strategy.decide import DEFAULT_WEIGHTS, NUM_CTX, SHORTLIST, brief, options_text, score, shortlist
from src.strategy.game import GameState
from src.strategy.intent import Intent
from src.strategy.moves import plans, situation_of

log = get_logger(__name__)

MAX_STEPS = 6  # lookups before it has to commit; a loop that never commits is a loop that hangs
MAX_SECONDS = 180.0  # the humans' turns are minutes long, but not unbounded
# The first live run, 2026-09-17: the model committed on step 1 without looking anything up, and
# explained a Rest by "o combate iminente no espaço do Mar" - no such space, no such combat, and
# it had read neither. Having tools is not the same as using them, and a prompt asking nicely is
# not a mechanism. It is sent back until it has read something.
MIN_LOOKUPS = 1

REFLECT_PROMPT = """You are Reachy, a small desktop robot playing Eldritch Horror (Fantasy Flight Games, 2013, base game only) as a real player. It is not your turn. The other players are taking theirs, and you have a few minutes to work out what YOUR investigator should be trying to do in the round that is coming.

You are not choosing a turn. A score chooses turns, out of the legal ones, and it is good at it. What it does not know is intent: which of two nearly equal turns serves the round you mean to have. That is what you are deciding.

You have one tool. Answer with {{"look_up": "your question"}} and you will be given passages from the rulebook, the reference guide, the FAQ and the component knowledge base, and then asked again. You must use it at least once before you commit: this time exists so that what you decide rests on something you read, not on something you remember. You have at most {steps} lookups.

When you are ready, answer with {{"intents": [...]}}, at most {max_intents} of them, each:
- "kind": "reach" (a space worth walking towards), "prefer" (an action worth taking) or "avoid" (a space not to end a turn on)
- "target": the exact name of a space on the board, or one of the actions: Travel, Prepare for Travel, Acquire Assets, Rest, Trade, Component
- "why": one short sentence, the reason a player would give
- "grounded_in": the words from the passage or the state that support it, close to how they were written. This is checked against what you actually read, and an intent that cites nothing you read is thrown away.
- "weight": 0 to 2, how strongly

Rules you are held to:
- A target that is not a space on this board or one of those actions is thrown away. Use the names exactly as they appear in the state you were given.
- Every intent needs a reason, and the reason has to be ABOUT that target. A card's bonus is a reason for an action that uses that card, and is a reason for nothing else. If a reason does not connect, do not write the intent - two that hold are worth more than four that do not.
- Never invent a rule, a card, a space or an effect. If you are not sure, look it up; if the lookup does not say, leave it out.
- Fewer is better. A round with two priorities has priorities; a round with six has none."""

REFLECT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "look_up": {"type": "string"},
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "target": {"type": "string"},
                    "why": {"type": "string"},
                    "grounded_in": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["kind", "target", "why"],
            },
        },
    },
}


@dataclass
class Thinking:
    """What one round's thinking came to, and everything it did on the way."""

    intents: tuple[Intent, ...] = ()
    refused: tuple[str, ...] = ()
    lookups: tuple[str, ...] = ()
    steps: int = 0
    seconds: float = 0.0
    stopped: str = ""  # "committed" | "out of steps" | "out of time" | "cancelled" | "no model"
    round: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "intents": [i.record() for i in self.intents],
            "refused": list(self.refused),
            "lookups": list(self.lookups),
            "steps": self.steps,
            "seconds": round(self.seconds, 1),
            "stopped": self.stopped,
        }

    def describe(self) -> str:
        if not self.intents:
            return f"nothing settled this round ({self.stopped})"
        return "; ".join(i.describe() for i in self.intents)


def _read(text: str) -> dict[str, Any]:
    """The model's JSON out of whatever it wrapped it in."""
    blob = (text or "").strip()
    if not blob.startswith("{"):
        match = re.search(r"\{.*\}", blob, re.S)
        blob = match.group(0) if match else ""
    try:
        data = json.loads(blob) if blob else {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _turns_it_will_choose_between(game: GameState) -> str:
    """The shortlist the score would pick from right now, so the thinking is about real turns.

    Without this the loop decides in the abstract and its priorities land on turns that are not
    on the table. The board moves before the round starts, so this is the shape of the choice,
    not the choice itself.
    """
    state = game.robot_investigator
    if state is None:
        return ""
    situation = situation_of(game, state)
    legal = plans(game, situation)
    if not legal:
        return ""
    kept = shortlist([score(game, situation, plan, DEFAULT_WEIGHTS) for plan in legal], SHORTLIST)
    return options_text(kept)


class Reflection:
    """One round's thinking. Built when the round ends, cancelled when the next one starts."""

    def __init__(
        self,
        game: GameState,
        *,
        language: str = "en-US",
        look_up: Callable[[str], str] | None = None,
        chat_fn: Callable[..., Any] = chat,
        max_steps: int = MAX_STEPS,
        max_seconds: float = MAX_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.game = game
        self.language = language
        self.look_up = look_up if look_up is not None else _rules_lookup
        self.chat_fn = chat_fn
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.clock = clock
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """The round has started: stop thinking, whatever it was in the middle of."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> Thinking:
        started = self.clock()
        thinking = Thinking(round=self.game.round)
        state_text = brief(self.game, situation_of(self.game, self.game.robot_investigator))
        turns = _turns_it_will_choose_between(self.game)
        if turns:
            state_text += "\n\nThe kind of turn the score will be choosing between:\n" + turns
        messages = [
            {
                "role": "system",
                "content": REFLECT_PROMPT.format(
                    steps=self.max_steps, max_intents=intents_module.MAX_INTENTS
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{state_text}\n\nWhat should my investigator be trying to do in the round "
                    f"that is coming? Answer with JSON only: either look_up, or intents. Write "
                    f"every reason in {language_name(self.language)}, keeping the game terms in "
                    f"English as printed."
                ),
            },
        ]
        lookups: list[str] = []
        read: list[str] = [state_text]
        proposed: list[Any] = []
        while True:
            if self.cancelled:
                thinking.stopped = "cancelled"
                break
            if self.clock() - started > self.max_seconds:
                thinking.stopped = "out of time"
                break
            if thinking.steps >= self.max_steps:
                thinking.stopped = "out of steps"
                break
            thinking.steps += 1
            try:
                reply = self.chat_fn(messages, num_ctx=NUM_CTX, think=False, format=REFLECT_SCHEMA)
            except LLMError as exc:
                log.warning("reflect: the model did not answer (%s)", exc)
                thinking.stopped = "no model"
                break
            data = _read(reply.text)
            if data.get("intents"):
                if len(lookups) < MIN_LOOKUPS:
                    # Committing without reading anything is the failure this loop exists to
                    # answer, so it is not allowed to: it is sent back to look something up.
                    log.info("reflect: committed without reading anything; sending it back")
                    messages = messages + [
                        {"role": "assistant", "content": reply.text},
                        {
                            "role": "user",
                            "content": (
                                "Not yet. Look something up first, with look_up, and base your "
                                "intents on what comes back. Answer with look_up only."
                            ),
                        },
                    ]
                    continue
                proposed = list(data["intents"])
                thinking.stopped = "committed"
                break
            question = " ".join(str(data.get("look_up") or "").split())
            if not question:
                # Neither a question nor a commitment. Asking again for the same thing twice
                # is how a loop spends a whole round saying nothing, so it is told to commit.
                messages = messages + [
                    {"role": "assistant", "content": reply.text},
                    {"role": "user", "content": "Answer with intents now, even if it is only one."},
                ]
                continue
            lookups.append(question)
            log.info("reflect: looking up %r", question[:80])
            try:
                found = self.look_up(question)
            except Exception as exc:  # a lookup failing must not end the round's thinking
                log.warning("reflect: the lookup failed (%s)", exc)
                found = "Nothing came back for that."
            read.append(found)
            messages = messages + [
                {"role": "assistant", "content": reply.text},
                {
                    "role": "user",
                    "content": (
                        f"{found}\n\nNow either look_up something else or answer with intents."
                    ),
                },
            ]
        # Checked against what it actually read, not against what it says it knows.
        thinking.intents, thinking.refused = intents_module.verify(proposed, self.game, tuple(read))
        thinking.lookups = tuple(lookups)
        thinking.seconds = self.clock() - started
        log.info(
            "reflect: round %d, %d step(s) in %.0fs (%s): %s",
            thinking.round,
            thinking.steps,
            thinking.seconds,
            thinking.stopped,
            thinking.describe(),
        )
        return thinking


def _rules_lookup(question: str) -> str:
    """The rulebook and the knowledge base, as passages. Imported late: Qdrant is not always up."""
    from src.speech.eleven_agent import rules_lookup

    found = rules_lookup(question)
    return str(found.get("passages") or "Nothing came back for that.")


@dataclass
class Thinker:
    """Runs one :class:`Reflection` at a time, on its own thread, and holds what it decided.

    ``start`` when the round ends, ``take`` when the robot is about to play. Starting a new one
    cancels whatever was still running: the state it was thinking about has moved on.
    """

    game: GameState
    language: str = "en-US"
    chat_fn: Callable[..., Any] = chat
    on_done: Callable[[Thinking], None] | None = None
    background: bool = True
    _running: Reflection | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    last: Thinking | None = field(default=None, init=False)

    def start(self) -> None:
        self.cancel()
        if self.game.robot_investigator is None:
            return  # nothing to think about until the robot knows which investigator is its own
        reflection = Reflection(self.game, language=self.language, chat_fn=self.chat_fn)
        self._running = reflection

        def work() -> None:
            thinking = reflection.run()
            if reflection.cancelled:
                return
            self.last = thinking
            if self.on_done is not None:
                self.on_done(thinking)

        if not self.background:
            work()
            return
        self._thread = threading.Thread(target=work, name="reflect", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        if self._running is not None:
            self._running.cancel()
        self._running = None

    def take(self) -> tuple[Intent, ...]:
        """The priorities for the round about to be played, if the thinking finished in time."""
        return self.last.intents if self.last is not None else ()


__all__ = [
    "MAX_SECONDS",
    "MAX_STEPS",
    "REFLECT_PROMPT",
    "REFLECT_SCHEMA",
    "Reflection",
    "Thinker",
    "Thinking",
]
