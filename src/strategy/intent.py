"""What the thinking between rounds is allowed to say, and how the score reads it.

    kept, refused = verify(proposed, game)     # only what the board and the rules support
    scored = score(game, situation, plan, intents=kept)

The reasoning model's failure mode was measured and named on 2026-09-17: **the facts it quotes
are real and the links between them are invented.** It justified a turn by the Protective
Amulet's +1 Will, which is printed on the card and was in its prompt, welded to a Prepare for
Travel, which has nothing to do with it. That is why it no longer chooses turns
(``decide.MODEL_PICKS``) and why what it *is* allowed to hand the score is this, and not prose.

An intent is deliberately small and deliberately checkable:

* ``reach <space>``  - that space is worth walking towards this round.
* ``prefer <action>`` - that action is worth taking this round.
* ``avoid <space>``  - do not end the turn there.

Three kinds, each with a target the code can look up in the map table or the action list. An
intent naming a space that is not on the board, or an action that is not one of the six printed
on the reference card, is refused here and never reaches a score - so a link the model invented
cannot become a move. What is *not* checked is whether the intent is any good; that is a matter
of opinion and belongs to the owner, which is why every one of them is logged with the sentence
that proposed it.

The weight is clamped, and the score's own term for it is one point (``Weights.intent``). At
most, an intent moves a turn by as much as one Monster standing on the space it ends on. It
re-ranks turns the score had nearly level; it cannot overturn "there is a fight waiting there".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from src.logger import get_logger
from src.strategy import map_graph, moves
from src.strategy.game import GameState

log = get_logger(__name__)

REACH, PREFER, AVOID = "reach", "prefer", "avoid"
KINDS = (REACH, PREFER, AVOID)
MAX_WEIGHT = 2.0  # one intent is worth at most one Monster in the way
MAX_INTENTS = 4  # a round's thinking that comes back with ten priorities has none
# How many words of ``grounded_in`` have to appear in something the loop actually read before the
# intent counts as grounded. Measured against the first live run on 2026-09-17, where the model
# committed without looking anything up and explained a Rest by "o combate iminente no espaço do
# Mar" - there is no such space, there was no combat, and it had read neither. Two content words
# is low enough that a paraphrase survives and high enough that a story does not.
MIN_CITED_WORDS = 2


@dataclass(frozen=True)
class Intent:
    """One thing the robot decided between rounds, in a shape the score can act on."""

    kind: str
    target: str
    why: str = ""  # the loop's own words, for the log and for the owner to argue with
    weight: float = 1.0
    grounded_in: str = ""  # what it looked up before saying this

    def describe(self) -> str:
        return f"{self.kind} {self.target} ({self.why})" if self.why else f"{self.kind} {self.target}"

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "why": self.why,
            "weight": round(self.weight, 2),
            "grounded_in": self.grounded_in,
        }


def _space(target: str) -> str:
    """The board's own name for a space, or "" when the board has no such space.

    Spaces are named both by number and by city ("5", "San Francisco"), and a model writes
    "space 5" and "Space 5" and " 5 ". The map table is the authority on which exist.
    """
    said = str(target or "").strip()
    for prefix in ("space ", "the space ", "espaço ", "espaco "):
        if said.lower().startswith(prefix):
            said = said[len(prefix) :].strip()
    if not said:
        return ""
    for known in map_graph.all_spaces():
        if known.lower() == said.lower():
            return known
    return ""


def _action(target: str) -> str:
    """The action key for a name the model wrote, or "" when it is not one of the six."""
    said = str(target or "").strip().lower().replace(" ", "_")
    if said in moves.BY_KEY:
        return said
    for key, rule in moves.BY_KEY.items():
        if rule.name.lower() == str(target or "").strip().lower():
            return key
    return ""


def _content_words(text: str) -> set[str]:
    """Words worth matching on: four letters or more, lower-cased, accents removed."""
    flat = "".join(
        c for c in unicodedata.normalize("NFD", (text or "").lower()) if unicodedata.category(c) != "Mn"
    )
    return {w for w in re.findall(r"[a-z0-9']+", flat) if len(w) >= 4}


def _cited(grounded_in: str, sources: tuple[str, ...]) -> bool:
    """Does what it says it read actually appear in something it read?

    This is the check that answers the failure the whole loop exists for: the facts are real and
    the link is invented. A model that has to point at the passage it used can still paraphrase,
    and cannot make the passage up - "o combate iminente no espaço do Mar" shares nothing with
    any text it was given, and is refused.
    """
    said = _content_words(grounded_in)
    if len(said) < MIN_CITED_WORDS:
        return False
    return any(len(said & _content_words(source)) >= MIN_CITED_WORDS for source in sources)


def verify(
    proposed: list[Any], game: GameState, sources: tuple[str, ...] = ()
) -> tuple[tuple[Intent, ...], tuple[str, ...]]:
    """Keep the intents the board and the reference card support; say why the rest were refused.

    ``proposed`` is whatever the loop came back with - dictionaries straight out of a model's
    JSON, or :class:`Intent` objects. Nothing here trusts the target: a space is looked up in the
    map table and an action in the printed list, and one that is not there is refused with the
    words that were said, so a refusal is readable in the log rather than silent.

    ``sources`` are the texts the loop actually read - the state it was given and what each of
    its lookups returned. When they are supplied, an intent has to cite one of them: see
    :func:`_cited`. When they are not (a caller checking a target by hand), the citation is not
    asked for.
    """
    kept: list[Intent] = []
    refused: list[str] = []
    for item in proposed or []:
        data = item.record() if isinstance(item, Intent) else dict(item or {})
        kind = str(data.get("kind", "")).strip().lower()
        said = str(data.get("target", "")).strip()
        why = " ".join(str(data.get("why", "")).split())
        if kind not in KINDS:
            refused.append(f"{kind or '(nothing)'} {said}: not one of {', '.join(KINDS)}")
            continue
        if not why:
            # A priority with no reason cannot be argued with, and arguing with them is what the
            # owner is for. It costs the model one short sentence to keep it.
            refused.append(f"{kind} {said}: no reason given")
            continue
        target = _action(said) if kind == PREFER else _space(said)
        if not target:
            refused.append(
                f"{kind} {said!r}: "
                + ("not an action on the reference card" if kind == PREFER else "not a space on this board")
            )
            continue
        if kind == AVOID and target == (game.robot_investigator.space if game.robot_investigator else ""):
            # Standing there already is not a reason to score every turn down by the same amount.
            refused.append(f"{kind} {target}: it is standing there now")
            continue
        grounded_in = " ".join(str(data.get("grounded_in", "")).split())
        if sources and not _cited(grounded_in, sources):
            refused.append(
                f"{kind} {target}: {grounded_in!r} is not in anything it read"
                if grounded_in
                else f"{kind} {target}: nothing cited"
            )
            continue
        try:
            weight = max(0.0, min(MAX_WEIGHT, float(data.get("weight", 1.0))))
        except (TypeError, ValueError):
            weight = 1.0
        kept.append(Intent(kind, target, why, weight, grounded_in))
    if len(kept) > MAX_INTENTS:
        refused.extend(f"{i.kind} {i.target}: more than {MAX_INTENTS} priorities in one round" for i in kept[MAX_INTENTS:])
        kept = kept[:MAX_INTENTS]
    for line in refused:
        log.info("intent refused: %s", line)
    for intent in kept:
        log.info("intent kept: %s", intent.describe())
    return tuple(kept), tuple(refused)


def reach_spaces(intents: tuple[Intent, ...]) -> list[str]:
    """The spaces the round's thinking wants walked towards, for ``decide.goals``."""
    return [i.target for i in intents if i.kind == REACH]


def bonus(
    intents: tuple[Intent, ...], plan: Any, ends_at: str, unit: float
) -> tuple[float, list[str]]:
    """What the intents add to or take off one turn, and the words for it.

    ``reach`` is not handled here: it is handed to ``decide.goals`` instead, so that walking
    towards it is scored by the same distance term as walking towards a Gate. One idea, one
    number.
    """
    value, reasons = 0.0, []
    keys = {action.key for action in getattr(plan, "actions", ())}
    for intent in intents:
        if intent.kind == PREFER and intent.target in keys:
            value += unit * intent.weight
            reasons.append(f"this round I meant to {moves.BY_KEY[intent.target].name}: {intent.why}")
        elif intent.kind == AVOID and ends_at == intent.target:
            value -= unit * intent.weight
            reasons.append(f"this round I meant to keep off {intent.target}: {intent.why}")
    return value, reasons


__all__ = [
    "AVOID",
    "KINDS",
    "MAX_INTENTS",
    "MIN_CITED_WORDS",
    "MAX_WEIGHT",
    "PREFER",
    "REACH",
    "Intent",
    "bonus",
    "reach_spaces",
    "verify",
]
