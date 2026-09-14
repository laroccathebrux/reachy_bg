"""Is this utterance for the robot? Rule-based first pass, and the speak/stay-quiet log.

    decision = decide(text, language, seconds_since_robot_spoke=3.0, robot_turn=False)
    decision.addressed, decision.reason, decision.confidence

The rules, in order of confidence:

1. The robot's name (or what Whisper hears instead of it) appears: addressed.
2. A question or a short reply within ``FOLLOW_UP_WINDOW_S`` after the robot spoke: follow-up.
3. A question about the rules or the game while it is the robot's turn: probably for it.
4. A rules question to the table: answered when ``ANSWER_GAME_QUESTIONS`` is on (a
   knowledgeable player would), logged either way.
5. Everything else: stay quiet (people talk to each other most of the time).

Every decision is appended to ``data/game_logs/addressee.jsonl`` by :class:`TurnLogger`;
that file is the dataset for the learned classifier later.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import ANSWER_GAME_QUESTIONS, FOLLOW_UP_WINDOW_S, GAME_LOG_DIR, ROBOT_NAME_ALIASES

_INTERROGATIVES = {
    "pt-BR": (
        "quantas quantos quanto qual quais como quando onde por que porque o que oque pode posso podemos "
        "consigo dá da tem existe precisa devo deve".split()
    ),
    "en-US": "how what which when where why who can could may should do does is are am will would".split(),
}
_RULES_WORDS = {
    "pt-BR": "regra regras ação ações rodada fase turno carta cartas jogar jogo pode posso permitido".split(),
    "en-US": "rule rules action actions round phase turn card cards play allowed legal can".split(),
}
_SHORT_REPLY_MAX_WORDS = 3
_INTERJECTIONS = frozenset("hey hi ei oi ola e ok entao ta then so".split())


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))


def mentions_robot(text: str, aliases: tuple[str, ...] = ROBOT_NAME_ALIASES) -> bool:
    """The robot is named as a vocative: at the start or end of the sentence, or before a comma.

    Some aliases are ordinary English words ("rich"), so a match in the middle of a sentence
    ("the rich merchant") does not count.
    """
    words = _words(text)
    if not words:
        return False
    names = {_strip_accents(a) for a in aliases}
    candidates = {words[0], words[-1]}
    if len(words) > 1 and words[0] in _INTERJECTIONS:
        candidates.add(words[1])
    if names & candidates:
        return True
    before_punctuation = set(re.findall(r"([a-z0-9]+)\s*[,!?]", _strip_accents(text.lower())))
    return bool(names & before_punctuation)


def is_question(text: str, language: str) -> bool:
    if "?" in text:
        return True
    words = _words(text)
    return bool(words) and words[0] in _INTERROGATIVES.get(language, ())


def about_the_game(text: str, language: str) -> bool:
    words = set(_words(text))
    return any(w in words for w in _RULES_WORDS.get(language, ()))


def looks_like_echo(text: str, last_answer: str, *, min_shared: float = 0.6) -> bool:
    """True when most of the heard words are in the robot's last answer (its own voice)."""
    heard = _words(text)
    if len(heard) < 3 or not last_answer:
        return False
    spoken = set(_words(last_answer))
    shared = sum(1 for w in heard if w in spoken)
    return shared / len(heard) >= min_shared


@dataclass(frozen=True)
class Decision:
    addressed: bool
    reason: str
    confidence: float


def decide(
    text: str,
    language: str,
    *,
    seconds_since_robot_spoke: float | None = None,
    robot_turn: bool = False,
    follow_up_window_s: float = FOLLOW_UP_WINDOW_S,
    answer_game_questions: bool = ANSWER_GAME_QUESTIONS,
) -> Decision:
    if mentions_robot(text):
        return Decision(True, "name", 0.95)
    question = is_question(text, language)
    recent = seconds_since_robot_spoke is not None and seconds_since_robot_spoke <= follow_up_window_s
    if recent and question:
        return Decision(True, "follow_up_question", 0.8)
    if recent and len(_words(text)) <= _SHORT_REPLY_MAX_WORDS:
        return Decision(True, "follow_up_reply", 0.6)
    if robot_turn and question and about_the_game(text, language):
        return Decision(True, "robot_turn_question", 0.6)
    if question and about_the_game(text, language):
        return Decision(answer_game_questions, "game_question", 0.5)
    return Decision(False, "not_addressed", 0.7)


class TurnLogger:
    """Append one JSON line per heard utterance: the speak/stay-quiet dataset."""

    def __init__(self, path: Path | None = None):
        self.path = path or GAME_LOG_DIR / "addressee.jsonl"

    def log(self, decision: Decision, **fields: Any) -> dict[str, Any]:
        record = {"ts": round(time.time(), 3), **fields, **asdict(decision)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


__all__ = [
    "Decision",
    "TurnLogger",
    "decide",
    "looks_like_echo",
    "mentions_robot",
    "is_question",
    "about_the_game",
]
