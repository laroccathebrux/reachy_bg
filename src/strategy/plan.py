"""The plan for the whole game, written after the setup and revised as the game argues with it.

    plan = make_plan(game, language="pt-BR")   # right after the spoken setup
    print(plan.speak())                        # read it back to the table
    plan = revise(plan, game, ["doom fell to 12", "a Gate opened at Rome"])
    plan.history                               # what it dropped, and when

A turn decided on its own is a turn without memory: the robot would buy a card in round one,
walk the other way in round two and never be *going* anywhere. So after the setup it writes down
how this group means to win - the aim, what its own investigator is for, two to four priorities
in the order it would drop them, and what would make the plan wrong - and that text rides along
in every later turn decision (``src/strategy/decide.brief``).

The plan is **the robot's intention, not a rule and not a fact**. It says so wherever it is
shown, it is written from the state the table gave and nothing else, and it is revised rather
than defended: ``revise`` is given what has happened since, keeps what still holds and says in
one line what it dropped. Every version is kept in ``history``, which is the record of how the
robot's idea of the game changed - the thing Phase 5 learns from.

Community strategy entries are fetched here on purpose. The eleven entries in ``bg_knowledge``
are mostly about which investigators suit which Ancient One, which is advice about a *game*, not
about a turn; this is where they belong, labelled as one player's opinion with its source.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.llm.ollama_client import LLMError, chat
from src.llm.prompts import PLAN_SCHEMA, plan_messages
from src.logger import get_logger
from src.strategy.game import GameState

log = get_logger(__name__)

NUM_CTX = 8192  # explicit, always: Ollama reserves the model's native window otherwise
MAX_PRIORITIES = 4
MAX_WATCH = 3


@dataclass
class GamePlan:
    """How the robot means to win this game, and what would change its mind."""

    aim: str = ""
    my_role: str = ""
    priorities: tuple[str, ...] = ()
    watch_for: tuple[str, ...] = ()
    round_written: int = 0
    history: tuple[str, ...] = field(default_factory=tuple)  # "round 3: dropped ... because ..."

    @property
    def empty(self) -> bool:
        return not (self.aim or self.priorities)

    def speak(self) -> str:
        """The plan as the robot would read it out: the aim, its own part, the first priority.

        The lines were written by the model in the language of the table, so they are joined and
        never introduced: an English "First," glued in front of a Portuguese sentence is exactly
        the mixed-language answer this project does not allow.
        """
        parts = [self.aim, self.my_role, self.priorities[0] if self.priorities else ""]
        return " ".join(p.strip() for p in parts if p and p.strip())

    def as_text(self) -> str:
        """The plan as the prompts see it: short, labelled, one line each."""
        lines = [f"Aim: {self.aim}"] if self.aim else []
        if self.my_role:
            lines.append(f"My investigator's part: {self.my_role}")
        for index, item in enumerate(self.priorities, 1):
            lines.append(f"Priority {index}: {item}")
        for item in self.watch_for:
            lines.append(f"Watch for: {item}")
        return "\n".join(lines)

    def record(self) -> dict[str, Any]:
        return {
            "aim": self.aim,
            "my_role": self.my_role,
            "priorities": list(self.priorities),
            "watch_for": list(self.watch_for),
            "round_written": self.round_written,
            "history": list(self.history),
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> GamePlan:
        return cls(
            aim=str(data.get("aim", "")),
            my_role=str(data.get("my_role", "")),
            priorities=tuple(data.get("priorities", ())),
            watch_for=tuple(data.get("watch_for", ())),
            round_written=int(data.get("round_written", 0)),
            history=tuple(data.get("history", ())),
        )


def setup_brief(game: GameState, *, advice: list[dict[str, Any]] | None = None) -> str:
    """The game as it stands, for writing a plan: who is playing, against what, with what skills."""
    lines: list[str] = []
    if game.ancient_one is not None:
        lines.append(
            f"Ancient One: {game.ancient_one.name}, {game.ancient_one.title}. Doom is at "
            f"{game.doom} and moves towards 0, where it awakens. "
            f"{game.ancient_one.mysteries_to_win} Mysteries solved wins the game."
        )
    lines.append(f"Mystery in play: {game.mystery or 'not told to me yet'}.")
    lines.append(f"Round {game.round or 1}, {game.players} investigators.")
    for state in game.investigators:
        skills = ", ".join(f"{k.capitalize()} {v}" for k, v in state.sheet.skills.items())
        who = "me" if state.is_robot else state.controller
        carried = ", ".join(state.possessions) or "nothing"
        lines.append(
            f"- {state.name}, {state.sheet.occupation} ({who}), at {state.space}: "
            f"{state.health} Health, {state.sanity} Sanity, {skills}; carries {carried}."
        )
    board = [f"{p.space}: {p.name}" for p in game.board.on_board if p.name and p.space]
    if board:
        lines.append("Named pieces on the board: " + "; ".join(board) + ".")
    if advice:
        lines.append("Player advice (one player's opinion from a community guide, never a rule):")
        for item in advice:
            source = item.get("source_author") or item.get("source_kind") or "a player guide"
            text = " ".join(str(item.get("text", "")).split())
            lines.append(f"  - {text[:300]} [{source}]")
    return "\n".join(lines)


def advice_for(game: GameState, *, fetch: Callable[..., list[dict[str, Any]]] | None = None, limit: int = 3):
    """Community advice about this Ancient One and these investigators, or [] if Qdrant is away."""
    if fetch is None:
        from src.rag.retrieve import advice as fetch  # imported late: Qdrant is optional here
    ancient = game.ancient_one.name if game.ancient_one else ""
    who = ", ".join(i.name for i in game.investigators)
    try:
        return fetch(f"advice for playing {ancient} with {who}", limit=limit)
    except Exception as exc:  # a missing Qdrant must not stop the robot from having a plan
        log.warning("plan: no strategy entries (%s)", exc)
        return []


def _read(text: str) -> dict[str, Any]:
    blob = text.strip()
    if not blob.startswith("{"):
        match = re.search(r"\{.*\}", blob, re.S)
        blob = match.group(0) if match else ""
    try:
        return json.loads(blob) if blob else {}
    except json.JSONDecodeError:
        return {}


def _build(data: dict[str, Any], game: GameState, previous: GamePlan | None) -> GamePlan:
    priorities = tuple(str(p).strip() for p in data.get("priorities", []) if str(p).strip())
    watch = tuple(str(w).strip() for w in data.get("watch_for", []) if str(w).strip())
    history = tuple(previous.history) if previous else ()
    changed = str(data.get("changed", "")).strip()
    if changed:
        history = history + (f"round {game.round or 1}: {changed}",)
    return GamePlan(
        aim=str(data.get("aim", "")).strip(),
        my_role=str(data.get("my_role", "")).strip(),
        priorities=priorities[:MAX_PRIORITIES],
        watch_for=watch[:MAX_WATCH],
        round_written=game.round or 1,
        history=history,
    )


def make_plan(
    game: GameState,
    *,
    language: str = "en-US",
    advice: list[dict[str, Any]] | None = None,
    chat_fn: Callable[..., Any] = chat,
    think: bool = False,
) -> GamePlan:
    """Write the plan for this game from the state the setup left behind.

    Returns an empty plan when the model cannot be reached: the robot plays without one rather
    than not playing, and ``decide`` simply has one block less in its brief.
    """
    started = time.perf_counter()
    messages = plan_messages(setup_brief(game, advice=advice), language)
    try:
        reply = chat_fn(messages, num_ctx=NUM_CTX, think=think, format=PLAN_SCHEMA)
    except LLMError as exc:
        log.warning("plan: the model did not answer (%s); playing without a plan", exc)
        return GamePlan()
    plan = _build(_read(reply.text), game, None)
    log.info("plan: %s (%.1fs)", plan.aim or "nothing usable", time.perf_counter() - started)
    return plan


def revise(
    plan: GamePlan,
    game: GameState,
    changes: list[str],
    *,
    language: str = "en-US",
    advice: list[dict[str, Any]] | None = None,
    chat_fn: Callable[..., Any] = chat,
    think: bool = False,
) -> GamePlan:
    """Rewrite the plan against what has happened, keeping what still holds.

    With nothing to react to, or with no plan yet, the plan in force is returned untouched: a
    round that changed nothing is not a reason to spend a model call.
    """
    if plan.empty:
        return make_plan(game, language=language, advice=advice, chat_fn=chat_fn, think=think)
    if not changes:
        return plan
    messages = plan_messages(
        setup_brief(game, advice=advice), language, previous=plan.as_text(), changes="; ".join(changes)
    )
    try:
        reply = chat_fn(messages, num_ctx=NUM_CTX, think=think, format=PLAN_SCHEMA)
    except LLMError as exc:
        log.warning("plan: revision failed (%s); the plan in force stands", exc)
        return plan
    revised = _build(_read(reply.text), game, plan)
    if revised.empty:
        return plan
    log.info(
        "plan: revised in round %d (%s)", game.round or 1, revised.history[-1] if revised.history else ""
    )
    return revised


def changes_since(plan: GamePlan, game: GameState, before: dict[str, Any] | None) -> list[str]:
    """What has happened since the plan was written, in the words the robot would use.

    ``before`` is an earlier ``GameState.record()``; without one, only what the state itself
    makes obvious is reported. This is what decides whether a revision is worth a model call.
    """
    out: list[str] = []
    if before:
        old_doom, new_doom = before.get("doom"), game.doom
        if old_doom is not None and new_doom is not None and new_doom != old_doom:
            out.append(f"doom moved from {old_doom} to {new_doom}")
        if before.get("mystery") and game.mystery and before["mystery"] != game.mystery:
            out.append(f"the Mystery changed to {game.mystery}")
        if int(before.get("mysteries_solved", 0)) != game.mysteries_solved:
            out.append(f"{game.mysteries_solved} Mysteries are solved now")
        old_spaces = {i["name"]: i.get("space") for i in before.get("investigators", [])}
        for state in game.investigators:
            was = old_spaces.get(state.name)
            if was and was != state.space:
                out.append(f"{state.name} moved from {was} to {state.space}")
    for state in game.investigators:
        if state.health <= 2 or state.sanity <= 2:
            out.append(f"{state.name} is down to {state.health} Health and {state.sanity} Sanity")
    named = [f"{p.name} at {p.space}" for p in game.board.on_board if p.name and p.space]
    if named and plan.round_written != (game.round or 1):
        out.append("on the board now: " + "; ".join(named))
    return out


__all__ = [
    "GamePlan",
    "MAX_PRIORITIES",
    "MAX_WATCH",
    "advice_for",
    "changes_since",
    "make_plan",
    "revise",
    "setup_brief",
]
