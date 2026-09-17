"""Choosing a turn out of the legal ones, and saying why out loud.

    decision = decide(game)                    # the robot's own investigator
    print(decision.plan.describe())            # "Travel to Tokyo, then Acquire Assets at Tokyo"
    print(decision.reason)                     # one to three spoken sentences
    decision.questions                         # what it would ask the table before acting

``moves.plans`` returns every legal turn, and on an ordinary turn that is between fifty and two
hundred of them. Handing all of those to the model is the wrong shape of problem for it: the
list alone is thousands of prompt tokens, which on this Mac is the largest part of the wait, and
a model reading two hundred numbered lines picks worse, not better.

So the work is split the way each side turned out to be good at it. A **score** here throws away
what is plainly bad - Rest at full Health, walking away from the only Gate, buying with nothing
in the Reserve - ranks what is left, and **takes the top one**. The **model** is then handed
that turn, and the score's own words for why it won, and asked for the sentence a player would
say. It never chooses. If it cannot answer at all, the score says it itself in plainer words and
the turn is unchanged. The robot always plays.

That division was measured, not assumed (``scripts/decide_replay.py``, 2026-09-17). The model
agreed with the score in 21 of 24 decisions and then 13 of 16, **every disagreement made the
turn worse**, and it answered differently on the same situation in 3 of 8 cases. It is a bad
chooser and a good speaker. ``MODEL_PICKS`` and ``decide(model_picks=True)`` put it back in the
choice, which is how the measuring goes on being possible.

**The score is the decision, and it is meant to be argued with.** It is deliberately simple and
readable, every term named in ``Weights``, so that when the owner disagrees with a turn there is
one number to change. Every plan that was cut is kept in ``Decision.considered`` and logged: a
plan the owner would have played that the score threw away is the dataset that fixes the score,
and later the world model of Phase 5.

**Unknowns are not scored away.** A candidate that depends on something the robot cannot see -
which Monster that piece is, what a Reserve card costs - keeps its question, and the question
reaches the table as ``Decision.questions`` rather than becoming an assumption.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.llm.ollama_client import LLMError, chat
from src.llm.prompts import (
    DECISION_SCHEMA,
    NARRATION_SCHEMA,
    decision_messages,
    narration_messages,
)
from src.logger import get_logger
from src.strategy import intent as intents_module
from src.strategy import map_graph, moves
from src.strategy.game import GameState, InvestigatorState
from src.strategy.moves import Situation, TurnPlan
from src.strategy.reference import mystery

log = get_logger(__name__)

SHORTLIST = 8  # turns shown to the model; the rest are logged, not offered
# Does the model choose the turn, or only say it? Measured with scripts/decide_replay.py on
# 2026-09-17: qwen3.6:35b-mlx agreed with the score in 21 of 24 decisions and then 13 of 16, and
# every single time it disagreed the turn it took was worse - it dropped the Rest of a hurt
# investigator, it added an Acquire Assets against an empty Reserve, and it gave up a Trade that
# was only possible because another investigator was standing on that space. It also answered
# differently on the same situation in 3 of 8 cases, so the same table state does not produce
# the same turn twice. So it does not choose. It still writes the sentence, which is the half it
# is good at, and its worst failure now costs a clumsy sentence instead of a worse move.
# scripts/decide_replay.py passes model_picks=True to go on measuring the other arrangement.
MODEL_PICKS = False
MAX_EFFECT_CHARS = 220  # a card's printed text is quoted, not retold, but not in full either
NUM_CTX = 8192  # always explicit: Ollama otherwise reserves the model's native window

# Goes with the action rules, because the model kept explaining a choice by the Clues it would
# have to spend: a Clue rerolls a die, and no action in this box is paid for with one.
TEST_RULE = (
    "Skill test: roll that skill's dice, a 5 or 6 is a success, and 1 Clue may be spent to "
    "reroll 1 die. Clues are not a currency for actions (Reference Guide, Skill Tests)."
)


@dataclass(frozen=True)
class Weights:
    """What the score is made of. Every number here is an opinion, and meant to be argued with."""

    recover: float = 1.0  # per point of Health or Sanity a Rest would bring back
    hurt_bonus: float = 3.0  # extra for resting at 2 or less of either
    approach_goal: float = 1.5  # per step closer to the nearest Gate or Clue
    acquire: float = 1.5  # taking the Acquire Assets action with cards in the Reserve
    influence: float = 0.3  # per point of Influence, which is what that test rolls
    prepare: float = 0.6  # a ticket, when the nearest goal is more than one move away
    component: float = 0.7  # using an "Action:" a card prints is value already paid for
    trade: float = 0.4
    monster_ahead: float = 2.0  # per named Monster on the space the turn ends on
    unknown: float = 0.2  # per thing the plan depends on that the robot cannot see
    # What one priority decided between rounds is worth (src/strategy/intent.py). Deliberately
    # the same size as one Monster: it re-ranks turns the score had nearly level and cannot
    # overturn "there is a fight waiting on that space".
    intent: float = 1.0


DEFAULT_WEIGHTS = Weights()


@dataclass
class Scored:
    """One turn with its score and the words behind it."""

    plan: TurnPlan
    score: float
    reasons: tuple[str, ...] = ()

    def record(self) -> dict[str, Any]:
        return {"score": round(self.score, 2), "reasons": list(self.reasons), **self.plan.record()}


def goals(game: GameState, intents: tuple[Any, ...] = ()) -> dict[str, list[str]]:
    """The board features worth walking towards, by space, as far as anything has named them.

    Gates and Clues are what an investigator travels for: a Gate closed is doom held back, a Clue
    is what solves a Mystery. Only pieces somebody has actually named count - an unnamed piece is
    a question, and a question is not a destination.

    The active Mystery is a destination too, and the better one: "Rituals in the Wild" happens on
    4, 10, 21 and Tunguska, "Spawn of Yog-Sothoth" on Arkham. Those spaces are printed on the
    card and known from ``src/rag/cards/eldritch_base_mysteries.json``, so winning the game is
    something the robot can walk towards rather than wait for.
    """
    found: dict[str, list[str]] = {}
    for piece in game.board.on_board:
        name = (piece.name or "").lower()
        if not piece.space:
            continue
        if name.startswith("gate") or name.startswith("clue"):
            found.setdefault(piece.space, []).append(piece.name or "")
    active = mystery(game.mystery) if game.mystery else None
    if active is not None:
        for space in active.spaces:
            found.setdefault(space, []).append(f"the Mystery {active.name}")
    # A space the round's own thinking decided to walk towards counts like any other
    # destination, and is scored by the same distance term. One idea, one number.
    for space in intents_module.reach_spaces(intents):
        found.setdefault(space, []).append("where I decided to go this round")
    return found


def _goal_words(labels: list[str]) -> str:
    """ "a Gate", "the Mystery Rituals in the Wild" - what is actually waiting on that space."""
    words = []
    for label in labels:
        kind, _, rest = label.partition(":")
        if kind.lower() == "gate":
            words.append("a Gate")
        elif kind.lower() == "clue":
            words.append("a Clue")
        else:
            words.append(rest.strip() or label)
    return " and ".join(dict.fromkeys(words)) or "something worth reaching"


def _monsters_on(game: GameState, space: str) -> list[str]:
    return moves.monsters_at(game, space)


def score(
    game: GameState,
    situation: Situation,
    plan: TurnPlan,
    weights: Weights = DEFAULT_WEIGHTS,
    intents: tuple[Any, ...] = (),
) -> Scored:
    """A readable number for one turn, and the reasons that made it.

    ``intents`` are the priorities the robot settled on between rounds, already checked against
    the board by ``src.strategy.intent.verify``. They are bounded on purpose: see that module.
    """
    value, reasons = 0.0, []
    goal_labels = goals(game, intents)
    goal_spaces = list(goal_labels)
    before = map_graph.nearest(situation.space, goal_spaces)[1] if goal_spaces else None
    ends_at = plan.ends_at

    if goal_spaces:
        after = map_graph.nearest(ends_at, goal_spaces)[1]
        if before is not None and after is not None and after != before:
            step = (before - after) * weights.approach_goal
            value += step
            where = map_graph.nearest(ends_at, goal_spaces)[0]
            reasons.append(
                f"{'closer to' if step > 0 else 'further from'} {where}, "
                f"where {_goal_words(goal_labels.get(where or '', []))} is"
            )

    for action in plan.actions:
        if action.key == moves.REST:
            missing = max(situation.max_health - situation.health, 0) + max(
                situation.max_sanity - situation.sanity, 0
            )
            value += missing * weights.recover
            if situation.health <= 2 or situation.sanity <= 2:
                value += weights.hurt_bonus
                reasons.append("Health or Sanity is low enough to be the next thing that kills it")
            elif missing:
                reasons.append(f"recovers {min(missing, 2)} of the {missing} points it is missing")
        elif action.key == moves.ACQUIRE:
            if game.reserve:
                sheet = game.by_name(situation.name)
                influence = sheet.sheet.influence if sheet else 0
                value += weights.acquire + influence * weights.influence
                reasons.append(f"the Reserve has {len(game.reserve)} cards and Influence is {influence}")
        elif action.key == moves.PREPARE:
            if before is not None and before > 1:
                value += weights.prepare
                reasons.append("a ticket is worth having while the nearest Gate is more than one move away")
        elif action.key == moves.COMPONENT:
            value += weights.component
            reasons.append(f"{action.detail} prints an Action that costs nothing to use")
        elif action.key == moves.TRADE:
            value += weights.trade

    monsters = _monsters_on(game, ends_at)
    if monsters:
        hurt = 1.0 + (situation.max_health - situation.health) / max(situation.max_health, 1)
        value -= weights.monster_ahead * len(monsters) * hurt
        reasons.append(f"ends the turn with {', '.join(monsters)} on the space, which means a fight")

    meant, said = intents_module.bonus(intents, plan, ends_at, weights.intent)
    value += meant
    reasons.extend(said)

    value -= weights.unknown * len(plan.unknowns)
    return Scored(plan, value, tuple(reasons))


def shortlist(scored: list[Scored], limit: int = SHORTLIST) -> list[Scored]:
    """The best turns, but never all of one kind: one plan of each first action survives first.

    A shortlist of eight Travels is a shortlist of one idea. Keeping the best plan of every
    first action means the model still sees "stay and buy" next to "go to Tokyo", whatever the
    score thinks of them.
    """
    best = sorted(scored, key=lambda s: -s.score)
    kept: list[Scored] = []
    seen_keys: set[str] = set()
    for item in best:
        key = item.plan.actions[0].key if item.plan.actions else ""
        if key not in seen_keys:
            seen_keys.add(key)
            kept.append(item)
    for item in best:
        if len(kept) >= limit:
            break
        if item not in kept:
            kept.append(item)
    return sorted(kept, key=lambda s: -s.score)[:limit]


def _effect(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_EFFECT_CHARS else text[:MAX_EFFECT_CHARS].rsplit(" ", 1)[0] + " ..."


def brief(
    game: GameState,
    situation: Situation,
    *,
    advice: list[dict[str, Any]] | None = None,
    plan: Any = None,
) -> str:
    """The state of the game in words, small enough to be cheap to read.

    Only what bears on the decision: the Ancient One and doom, the Mystery as it was told, the
    investigator's own sheet and what it carries, who else is where, what is on the spaces that
    matter, and what the Reserve cards do.
    """
    state = game.by_name(situation.name)
    sheet = state.sheet if state else None
    lines: list[str] = []
    if game.ancient_one is not None:
        doom = game.doom if game.doom is not None else game.ancient_one.starting_doom
        lines.append(
            f"Ancient One: {game.ancient_one.name}. Doom is at {doom} and moves towards 0; "
            f"at 0 {game.ancient_one.name} awakens, so {doom} is how much time is left "
            f"(it started at {game.ancient_one.starting_doom})."
        )
    active = mystery(game.mystery) if game.mystery else None
    if active is not None:
        where = f" It is at {', '.join(active.spaces)}." if active.spaces else ""
        lines.append(f"Mystery: {active.name}. To solve it: {active.requirement}{where}")
    else:
        lines.append(f"Mystery: {game.mystery or 'not told to me yet'}.")
    lines.append(f"Omen: {game.omen or 'not told to me yet'}.")
    lines.append(f"Round {game.round or 1}, Action Phase, {game.players} investigators in play.")
    if sheet is not None:
        skills = ", ".join(f"{k.capitalize()} {v}" for k, v in sheet.skills.items())
        lines.append(
            f"I play {sheet.name}, {sheet.occupation}, at {situation.space}: "
            f"{situation.health}/{sheet.health} Health, {situation.sanity}/{sheet.sanity} Sanity, "
            f"{situation.clues} Clue(s), {skills}."
        )
    tickets = []
    if situation.train_tickets:
        tickets.append(f"{situation.train_tickets} Train")
    if situation.ship_tickets:
        tickets.append(f"{situation.ship_tickets} Ship")
    lines.append("Tickets: " + (" and ".join(tickets) if tickets else "none") + ".")
    if situation.possessions:
        lines.append("I carry:")
        for card in situation.possessions:
            effect = moves.card_effect(card)
            lines.append(f"  - {card}: {_effect(effect) if effect else 'effect unknown to me'}")
    if situation.conditions:
        lines.append("Conditions on me: " + ", ".join(situation.conditions) + ".")
    others = [i for i in game.investigators if i.name != situation.name]
    if others:
        lines.append(
            "Other investigators: " + "; ".join(f"{i.name} ({i.controller}) at {i.space}" for i in others)
        )
    if game.reserve:
        lines.append("Reserve:")
        for card in game.reserve:
            effect = moves.card_effect(card)
            lines.append(f"  - {card}: {_effect(effect) if effect else 'effect unknown to me'}")
    board: dict[str, list[str]] = {}
    for piece in game.board.on_board:
        if not piece.space or not piece.name:
            continue
        kind, _, label = (piece.name or "").partition(":")
        kind = kind.strip().lower()
        if kind == "investigator":
            continue  # the investigators are listed above, by name and controller
        if kind == "gate":
            board.setdefault(piece.space, []).append("a Gate")
        elif kind == "clue":
            board.setdefault(piece.space, []).append("a Clue")
        elif kind == "monster":
            board.setdefault(piece.space, []).append(f"the Monster {label.strip() or 'nobody named'}")
        else:
            board.setdefault(piece.space, []).append(piece.name)
    if board:
        lines.append(
            "On the board: "
            + "; ".join(f"{space} has {' and '.join(what)}" for space, what in board.items())
            + ". Every other space is empty as far as I can see."
        )
    unnamed = moves.unnamed_pieces_at(game, situation.space)
    if unnamed:
        lines.append(f"There are {unnamed} piece(s) on my space that nobody has named yet.")
    if plan is not None and not getattr(plan, "empty", True):
        lines.append(
            "My plan for this game (my own intention, written after the setup, not a rule):\n"
            + plan.as_text()
        )
    if advice:
        lines.append("Player advice (opinion from a community guide, not a rule):")
        for item in advice:
            source = item.get("source_author") or item.get("source_kind") or "a player guide"
            lines.append(f"  - {_effect(str(item.get('text', '')))} [{source}]")
    return "\n".join(lines)


def options_text(kept: list[Scored]) -> str:
    """The shortlist as the model sees it: a number, the turn, and what it depends on."""
    lines = []
    for index, item in enumerate(kept, 1):
        line = f"{index}) {item.plan.describe()}"
        if item.plan.unknowns:
            line += " [depends on: " + "; ".join(item.plan.unknowns) + "]"
        lines.append(line)
    return "\n".join(lines)


def rules_of(kept: list[Scored]) -> str:
    """The printed rule of each action on the shortlist, in the reference card's own words.

    Without this the model explains its choice with a rule it half remembers: asked live, it
    said Acquire Assets would be paid for with Clues, which is a test of Influence. Six short
    lines are cheaper than one invented rule said out loud at the table.
    """
    keys = {action.key for item in kept for action in item.plan.actions}
    lines = [f"  - {moves.BY_KEY[key].describe()}" for key in (a.key for a in moves.ACTIONS) if key in keys]
    if lines:
        lines.append(f"  - {TEST_RULE}")
    return "\n".join(lines)


@dataclass
class Decision:
    """What the robot will do this turn, why, and what it would like to ask first."""

    plan: TurnPlan | None
    reason: str
    ask: str = ""  # the model's own question, written in the language the table is speaking
    questions: tuple[str, ...] = ()  # everything the plan depends on, in English, for the log
    chosen_by: str = "score"  # "model" when the LLM picked it, "score" when it fell back
    narrated_by: str = "template"  # who wrote `reason`: "model", or "template" from the score
    considered: tuple[Scored, ...] = ()
    seconds: float = 0.0
    refusals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def acted(self) -> bool:
        return self.plan is not None

    def record(self) -> dict[str, Any]:
        return {
            "plan": None if self.plan is None else self.plan.record(),
            "reason": self.reason,
            "ask": self.ask,
            "questions": list(self.questions),
            "chosen_by": self.chosen_by,
            "narrated_by": self.narrated_by,
            "seconds": round(self.seconds, 2),
            "refusals": list(self.refusals),
            "considered": [s.record() for s in self.considered],
        }


def _fallback_reason(item: Scored) -> str:
    """What the score would say, when the model says nothing usable."""
    what = item.plan.describe()
    if item.reasons:
        return f"{what}. {item.reasons[0].capitalize()}."
    return f"{what}."


def decide(
    game: GameState,
    investigator: InvestigatorState | None = None,
    *,
    language: str = "en-US",
    limit: int = SHORTLIST,
    weights: Weights = DEFAULT_WEIGHTS,
    intents: tuple[Any, ...] = (),
    advice: list[dict[str, Any]] | None = None,
    plan: Any = None,
    think: bool = False,
    model_picks: bool = MODEL_PICKS,
    chat_fn: Callable[..., Any] = chat,
) -> Decision:
    """Pick this investigator's turn and explain it in one to three spoken sentences.

    The **score** picks the turn; the model is asked only to say it out loud. That split was
    measured rather than assumed - see ``MODEL_PICKS`` above and ``scripts/decide_replay.py`` -
    and ``model_picks=True`` puts the model back in the choice, which is how the measuring is
    kept honest.

    ``think`` is off by default because it was measured on this Mac and is not worth it: with the
    model warm, a turn takes 4 s without the reasoning trace and 114 s with it, and the long
    answer was the one that invented a rule. A table does not wait two minutes for a move.

    ``chat_fn`` is the model call, injected so the whole pipeline can be exercised without
    Ollama. When the map table has not been checked against the board yet
    (``map_graph.VERIFIED``), a turn that moves is refused and the refusal is the answer: a path
    that is not printed on the board would be a rule the robot invented.
    """
    started = time.perf_counter()
    state = investigator or game.robot_investigator
    if state is None:
        return Decision(None, "I do not know which investigator I am playing yet.")
    situation = moves.situation_of(game, state)
    blocked = tuple(moves.refusals(game, situation))
    all_plans = moves.plans(game, situation)
    if not map_graph.VERIFIED:
        all_plans = [p for p in all_plans if not any(a.key == moves.TRAVEL for a in p.actions)]
        blocked = blocked + (
            "the table of paths between the spaces has not been checked against the board yet, "
            "so I will not move a piece on it",
        )
    if not all_plans:
        return Decision(
            None,
            "I cannot take an action this turn.",
            chosen_by="rules",
            seconds=time.perf_counter() - started,
            refusals=blocked,
        )

    scored = [score(game, situation, plan, weights, intents) for plan in all_plans]
    kept = shortlist(scored, limit)
    questions = tuple(dict.fromkeys(u for item in kept for u in item.plan.unknowns))

    state_text = brief(game, situation, advice=advice, plan=plan)
    rules = rules_of(kept)
    if rules:
        state_text += "\nThe actions in the list, as the reference card prints them:\n" + rules
    chosen, reason, ask, by = kept[0], "", "", "score"
    said_by = "template"
    if model_picks:
        messages = decision_messages(state_text, options_text(kept), language)
        for attempt in (1, 2):
            try:
                reply = chat_fn(messages, num_ctx=NUM_CTX, think=think, format=DECISION_SCHEMA)
            except LLMError as exc:
                log.warning("decide: the model did not answer (%s); the score decides", exc)
                break
            picked, reason, ask = _read_reply(reply.text)
            if picked is not None and 1 <= picked <= len(kept):
                chosen, by, said_by = kept[picked - 1], "model", "model"
                break
            log.info(
                "decide: attempt %d gave %r, which is not one of the %d turns", attempt, picked, len(kept)
            )
            messages = messages + [
                {"role": "assistant", "content": reply.text},
                {
                    "role": "user",
                    "content": (
                        "That is not one of the turns. Answer again with choice between 1 and "
                        f"{len(kept)}."
                    ),
                },
            ]
            reason = ""
    else:
        # The turn is settled; the model is handed it, and the score's own words for why, and
        # asked for the sentence. A model that cannot answer costs a plainer sentence and never
        # a different turn, which is the whole point of doing it this way round.
        messages = narration_messages(
            state_text,
            chosen.plan.describe(),
            "; ".join(chosen.reasons),
            options_text(kept[1:]),
            language,
        )
        try:
            reply = chat_fn(messages, num_ctx=NUM_CTX, think=think, format=NARRATION_SCHEMA)
        except LLMError as exc:
            log.warning("decide: the model did not answer (%s); the score says it itself", exc)
        else:
            _, reason, ask = _read_reply(reply.text)
            if reason.strip():
                said_by = "model"

    if not reason.strip():
        reason = _fallback_reason(chosen)
        said_by = "template"
    asked = tuple(dict.fromkeys(([ask.strip()] if ask.strip() else []) + list(questions)))
    decision = Decision(
        chosen.plan,
        reason.strip(),
        ask=ask.strip(),
        questions=asked,
        chosen_by=by,
        narrated_by=said_by,
        considered=tuple(sorted(scored, key=lambda s: -s.score)),
        seconds=time.perf_counter() - started,
        refusals=blocked,
    )
    log.info(
        "decide: %s (chosen by the %s, said by the %s, %.1fs, %d plans, %d shown)",
        decision.plan.describe() if decision.plan else "nothing",
        by,
        said_by,
        decision.seconds,
        len(scored),
        len(kept),
    )
    return decision


def _read_reply(text: str) -> tuple[int | None, str, str]:
    """(choice, reason, ask) out of the model's JSON, tolerant of a stray sentence around it."""
    blob = text.strip()
    if not blob.startswith("{"):
        match = re.search(r"\{.*\}", blob, re.S)
        blob = match.group(0) if match else ""
    try:
        data = json.loads(blob) if blob else {}
    except json.JSONDecodeError:
        return None, "", ""
    choice = data.get("choice")
    try:
        choice = int(choice)
    except (TypeError, ValueError):
        choice = None
    return choice, str(data.get("reason") or ""), str(data.get("ask") or "")


__all__ = [
    "DEFAULT_WEIGHTS",
    "SHORTLIST",
    "Decision",
    "Scored",
    "TEST_RULE",
    "Weights",
    "brief",
    "decide",
    "goals",
    "options_text",
    "rules_of",
    "score",
    "shortlist",
]
