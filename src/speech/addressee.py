"""Is this utterance for the robot? Rule-based first pass, and the speak/stay-quiet log.

    decision = decide(text, language, seconds_since_robot_spoke=3.0, robot_turn=False)
    decision.addressed, decision.reason, decision.confidence

The rules, in order of confidence:

1. The robot's name (or what Whisper hears instead of it) appears: addressed.
1b. The investigator the robot is playing is named ("it is Lily Chen's turn"): addressed, and
   with full confidence when the sentence hands over the turn.
2. Another player is named as the vocative ("Bruno, what do you think?"): not for the robot.
3. A question or a short reply within ``FOLLOW_UP_WINDOW_S`` after the robot spoke: follow-up.
3b. The robot is missing something it can only be told and the sentence sounds like the
   briefing for it: a statement counts, and the robot stays quiet unless it learnt something.
3c. The speaker still holds the floor: they spoke to the robot less than ``FLOOR_WINDOW_S``
   ago and nobody else has spoken since, so this sentence continues what they were telling it -
   a statement counts, which is how a person explains a board.
4. A question about the rules or the game while it is the robot's turn: probably for it.
5. A rules question to the table: answered when ``ANSWER_GAME_QUESTIONS`` is on (a
   knowledgeable player would), logged either way.
6. With a single person at the table (``humans_present=1``), anything they say out loud is
   for the robot (second-person forms with more confidence).
7. Everything else: stay quiet (people talk to each other most of the time).

Every decision is appended to ``data/game_logs/addressee.jsonl`` by :class:`TurnLogger`;
that file is the dataset for the learned classifier later.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import (
    ANSWER_GAME_QUESTIONS,
    FLOOR_WINDOW_S,
    FOLLOW_UP_WINDOW_S,
    GAME_LOG_DIR,
    ROBOT_NAME_ALIASES,
)

# Single words that open a question. "o que" and "por que" are pairs and live below: split on
# whitespace they leave a bare "o", and then every Portuguese sentence starting with "o" - "o
# ancião é Azathoth", "O Mystery é esse" - was read as a question.
_INTERROGATIVES = {
    "pt-BR": frozenset(
        "quantas quantos quanto qual quais como quando onde quem porque oque pode posso podemos "
        "consigo da tem existe precisa devo deve".split()
    ),
    "en-US": frozenset(
        "how what which when where why who can could may should do does is are am will would".split()
    ),
}
_INTERROGATIVE_PAIRS = {
    "pt-BR": frozenset({("o", "que"), ("por", "que"), ("pra", "que"), ("para", "que"), ("de", "que")}),
    "en-US": frozenset({("how", "many"), ("how", "much"), ("what", "about")}),
}
_RULES_WORDS = {
    "pt-BR": (
        "regra regras ação ações rodada fase turno carta cartas jogar jogo permitido descansar viajar "
        "comprar lutar monstro monstros portal portais pista pistas investigador investigadores dado dados".split()
    ),
    "en-US": "rule rules action actions round phase turn card cards play allowed legal".split(),
}
_SHORT_REPLY_MAX_WORDS = 3
# Second-person forms: with a single known player at the table, "you" can only be the robot.
_SECOND_PERSON = {
    "pt-BR": frozenset("voce tu te ti contigo teu tua seu sua consegue sabe pode quer".split()),
    "en-US": frozenset("you your yours yourself".split()),
}
_INTERJECTIONS = frozenset("hey hi ei oi ola e ok entao ta then so".split())


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))


def _vocative(text: str, names: set[str]) -> bool:
    """One of ``names`` is used as a vocative: first or last word, after an interjection, or before a comma."""
    words = _words(text)
    if not words or not names:
        return False
    candidates = {words[0], words[-1]}
    if len(words) > 1 and words[0] in _INTERJECTIONS:
        candidates.add(words[1])
    if names & candidates:
        return True
    before_punctuation = set(re.findall(r"([a-z0-9]+)\s*[,!?]", _strip_accents(text.lower())))
    return bool(names & before_punctuation)


def mentions_robot(text: str, aliases: tuple[str, ...] = ROBOT_NAME_ALIASES) -> bool:
    """The robot is named as a vocative: at the start or end of the sentence, or before a comma.

    Some aliases are ordinary English words ("rich"), so a match in the middle of a sentence
    ("the rich merchant") does not count.
    """
    return _vocative(text, {_strip_accents(a.lower()) for a in aliases})


def mentions_person(text: str, names: Iterable[str]) -> bool:
    """Someone else at the table is named as the vocative (first names, accents ignored)."""
    people = {_strip_accents(n.strip().lower().split()[0]) for n in names if n and n.strip()}
    return _vocative(text, people)


# "It's Lily Chen's turn" is addressed to the robot when the robot is playing Lily Chen. The
# owner should not have to say the robot's name to hand it its own turn.
_TURN_PHRASES = {
    "pt-BR": (
        r"\b(?:e|eh)?\s*(?:a|sua|tua)?\s*vez\b",
        r"\bvez\s+d[oa]\b",
        r"\bpode\s+jogar\b",
        r"\bjoga\s+a[ií]\b",
        r"\bt[au]\s+na\s+sua\s+vez\b",
    ),
    "en-US": (
        r"\b(?:it'?s\s+)?(?:your|his|her|their)\s+turn\b",
        r"\bturn\s+now\b",
        r"\b\w+'s\s+turn\b",
        r"\byou'?re\s+up\b",
        r"\bgo\s+ahead\b",
    ),
}


def is_turn_call(text: str, language: str) -> bool:
    """Does the sentence hand the turn to somebody?"""
    flat = _strip_accents(text.lower())
    patterns = _TURN_PHRASES.get(language) or _TURN_PHRASES["en-US"]
    return any(re.search(p, flat) for p in patterns)


def names_investigator(text: str, investigator_name: str) -> bool:
    """Is this investigator named anywhere in the sentence?

    Unlike a vocative, the name can sit in the middle ("it is Lily Chen's turn now"), and any
    part of it counts, because that is how people speak: "Lily", "Chen", "Lily Chen".
    """
    if not investigator_name.strip():
        return False
    words = set(_words(text))
    parts = {_strip_accents(p.lower()) for p in investigator_name.split() if len(p) > 2}
    return bool(parts & words)


_LEADING = frozenset("e ai entao mas ok ta hey ei oi ola and so but then well".split())


def is_question(text: str, language: str) -> bool:
    """A question mark, or an interrogative as the first word (after "e", "então", "so", "and"...)."""
    if "?" in text:
        return True
    words = _words(text)
    while len(words) > 1 and words[0] in _LEADING:
        words = words[1:]
    if not words:
        return False
    if words[0] in _INTERROGATIVES.get(language, frozenset()):
        return True
    pair = tuple(words[:2])
    return len(pair) == 2 and pair in _INTERROGATIVE_PAIRS.get(language, frozenset())


# Somebody saying out loud that they are about to read something, or asking the robot to hold
# on because they have not finished. The answer to one of these is silence, not a sentence: a
# reader pauses to breathe, VAD_SILENCE_MS is 600 ms, and the robot answered into the gap.
#
# The agent has a ``listening`` tool for exactly this and never called it once in a whole
# session, so the mechanism sat idle while the prompt got the visible half right. This is the
# same judgement made locally, on the transcript the gate already has, before anything reaches
# the cloud - and a rule that fires here costs no tokens and cannot be forgotten by a model.
# Matched against accent-stripped, lower-cased text (``_strip_accents``), so the patterns carry
# no accents either: "peraí" arrives here as "perai" and r"\bpera[ií]\b" would never fire.
_READING_AHEAD = {
    "pt-BR": (
        r"\b(?:vou|vamos|deixa\s+eu|deixe\s+me|me\s+deixa)\s+ler\b",
        r"\b(?:estou|to|tou)\s+(?:lendo|falando|explicando|terminando)\b",
        r"\bdeixa\s+eu\s+(?:falar|terminar|acabar|explicar)\b",
        r"\bnao\s+(?:me\s+)?interromp",
        r"\bso\s+um\s+(?:minuto|segundo|instante|momento|pouco|pouquinho)\b",
        r"\b(?:espera|esperai|perai|aguarda)\b",
    ),
    "en-US": (
        r"\b(?:let\s+me|i'?ll|i\s+am\s+going\s+to|i'?m\s+going\s+to)\s+read\b",
        r"\bi'?m\s+(?:reading|talking|explaining|still\s+\w+ing)\b",
        r"\bi\s+am\s+(?:reading|talking|explaining)\b",
        r"\blet\s+me\s+(?:finish|talk|explain)\b",
        r"\bdon'?t\s+interrupt\b",
        r"\b(?:hold\s+on|hang\s+on|one\s+(?:second|moment|minute))\b",
        r"\bgive\s+me\s+a\s+(?:second|moment|minute)\b",
        r"\bwait\b",
    ),
}


def announces_reading(text: str, language: str) -> bool:
    """Is the speaker telling the robot to wait while they read or finish talking?

    A question never counts, however it is phrased: "espera, quantos dados eu rolo?" is a
    question with "espera" in front of it, and holding the floor on it would leave the table
    waiting for an answer that never comes. That is the same line the agent's prompt draws -
    "somebody announcing what they are about to do is not a question" - drawn here instead.
    """
    if is_question(text, language):
        return False
    flat = _strip_accents(text.lower())
    patterns = _READING_AHEAD.get(language) or _READING_AHEAD["en-US"]
    return any(re.search(p, flat) for p in patterns)


_GAME_TERMS = frozenset(
    "doom omen gate gates mystery mysteries clue clues monster monsters investigator investigators ancient "
    "asset assets spell spells artifact artifacts condition conditions encounter encounters mythos rumor "
    "expedition travel rest trade acquire health sanity delayed detained lore influence observation strength "
    "will token tokens dice die roll rolls reroll focus ticket tickets space city sea wilderness action actions "
    "phase round turn card cards rule rules setup win lose awaken awakens".split()
)


def needs_rules(text: str, language: str) -> bool:
    """True when the utterance is about the game (rules lookup); False for greetings and banter."""
    words = set(_words(text))
    if words & _GAME_TERMS:
        return True
    return about_the_game(text, language)


def second_person(text: str, language: str) -> bool:
    words = set(_words(text))
    return bool(words & _SECOND_PERSON.get(language, frozenset()))


def about_the_game(text: str, language: str) -> bool:
    words = set(_words(text))  # accent-stripped, so compare stripped keywords too
    return any(_strip_accents(w) in words for w in _RULES_WORDS.get(language, ()))


def _same_word(heard: str, spoken: set[str], stems: dict[str, set[int]]) -> bool:
    """Exact match, or the same first four letters and about the same length ("deixa" ~ "deixe").

    The length check keeps "posso" from matching "possível" through the shared stem.
    """
    if heard in spoken:
        return True
    lengths = stems.get(heard[:4], set()) if len(heard) >= 5 else set()
    return any(abs(n - len(heard)) <= 1 for n in lengths)


def looks_like_echo(text: str, last_answer: str, *, min_shared: float = 0.6) -> bool:
    """True when the heard words are mostly the robot's own recent words (its echo at the mic).

    Only content words (four letters or more) count: function words ("eu", "não", "que") are
    shared by any two sentences in the same language. Echo arrives distorted, so longer words
    also match on their first four letters. With fewer than two content words the text is
    echo only when every word was spoken.
    """
    heard = _words(text)
    if not heard or not last_answer:
        return False
    spoken = set(_words(last_answer))
    stems: dict[str, set[int]] = {}
    for w in spoken:
        if len(w) >= 5:
            stems.setdefault(w[:4], set()).add(len(w))
    content = [w for w in heard if len(w) >= 4]
    if len(content) < 2:
        return all(_same_word(w, spoken, stems) for w in heard)
    shared = sum(1 for w in content if _same_word(w, spoken, stems))
    return shared / len(content) >= min_shared


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
    humans_present: int | None = None,
    follow_up_ok: bool = True,
    other_names: Iterable[str] = (),
    my_investigator: str = "",
    seconds_since_addressed: float | None = None,
    floor_window_s: float = FLOOR_WINDOW_S,
    setup_hint: bool = False,
) -> Decision:
    """Decide whether the utterance is for the robot.

    ``humans_present`` is how many people can be talked to besides the robot (None =
    unknown); ``follow_up_ok`` says whether the speaker is the one the robot was talking to
    (a stranger's cough right after an answer is not a follow-up); ``other_names`` are the
    other players at the table (the speaker excluded), so that naming one of them means the
    sentence is not for the robot.

    ``my_investigator`` is the investigator the robot is playing. Handing the robot its turn by
    naming its investigator - "it is Lily Chen's turn" - counts exactly like calling its own
    name, which is how people actually pass the turn at a table.

    ``seconds_since_addressed`` is how long ago this same voice last said something the robot
    took as its own, and it is what lets a person *tell* the robot something rather than only
    ask it. In the first live session the owner explained a card in four breaths - "Reachy, eu
    comprei uma carta", "chamada Deep Ones Attack", "quando ela entra em jogo...", "coloquei um
    Eldritch Token no espaço 18" - and only the breaths that happened to be questions got
    through; the rest were dropped as table talk. While the floor is open, a statement counts.
    Naming another player still ends it, because that sentence is plainly for them.
    """
    if mentions_robot(text):
        return Decision(True, "name", 0.95)
    if my_investigator and names_investigator(text, my_investigator):
        # The robot's own investigator was named. A turn call is unambiguous; anything else
        # naming it is still far more likely to be for the robot than not.
        if is_turn_call(text, language):
            return Decision(True, "my_turn", 0.95)
        return Decision(True, "my_investigator", 0.8)
    if mentions_person(text, other_names):
        return Decision(False, "other_person", 0.85)
    question = is_question(text, language)
    if is_turn_call(text, language) and (mentions_robot(text) or humans_present == 1):
        return Decision(True, "turn_call", 0.75)
    if humans_present == 1 and len(_words(text)) >= 2:
        # One person at the table: whatever they say out loud is for the robot (a second-person
        # form makes it certain, anything else is still far more likely than talking alone).
        reason = "second_person_solo" if second_person(text, language) else "solo"
        return Decision(True, reason, 0.6 if reason == "second_person_solo" else 0.55)
    recent = (
        follow_up_ok
        and seconds_since_robot_spoke is not None
        and seconds_since_robot_spoke <= follow_up_window_s
    )
    if recent and question:
        return Decision(True, "follow_up_question", 0.8)
    if recent and len(_words(text)) <= _SHORT_REPLY_MAX_WORDS:
        return Decision(True, "follow_up_reply", 0.6)
    if setup_hint and not question:
        # The robot is still missing something it can only be told, and this sentence carries
        # the words of a briefing. A person saying "o mistério atual é The Deep Ones Attack" at
        # a table where the robot does not know the Mystery is telling the robot. It costs a
        # local extraction and nothing in the cloud, and the robot only answers if it learnt
        # something, so being wrong here is quiet.
        return Decision(True, "setup_talk", 0.6)
    if seconds_since_addressed is not None and 0 <= seconds_since_addressed <= floor_window_s:
        # The same person is still talking to the robot: this is the rest of what they were
        # saying, whether or not it happens to be a question.
        return Decision(True, "holding_the_floor", 0.7)
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
    "mentions_person",
    "is_question",
    "about_the_game",
    "second_person",
    "needs_rules",
    "announces_reading",
]
