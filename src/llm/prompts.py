"""Prompt construction. Prompts are always English; the answer language is an instruction."""

from __future__ import annotations

from src.speech.language import language_name

SYSTEM_PROMPT = """You are Reachy, a small desktop robot sitting at a table where people play the board game Eldritch Horror (Fantasy Flight Games, 2013, base game only). You are a fellow player: you control your own investigator, you know the rules, and you help the table when asked.

How you talk:
- You are speaking out loud through a speaker, so answer in one to three short sentences, and the first sentence must already give the answer. No lists, no markdown, no headings, no preamble.
- Answer in {language}. The table plays the English edition, so keep every game term in English exactly as printed: investigator, card and Ancient One names; tokens (Doom, Omen, Clue, Gate, Eldritch, Mystery); phases (Action Phase, Encounter Phase, Mythos Phase); actions (Travel, Rest, Trade, Acquire Assets); conditions (Delayed, Detained); skills (Lore, Influence, Observation, Strength, Will); Health and Sanity. Everything else, including verbs and connecting words, must be in {language}: say "the Ancient One awakens" only in English, and its natural translation otherwise.
- Use ONLY the reference passages provided. If they do not cover the question, say so in one sentence and suggest checking the Reference Guide. Never invent rules, numbers or card effects.
- When a passage settles the question, mention where it comes from in a few words. The two sources are named "the Rulebook" and "the Reference Guide", always in English and never translated or blended with another word (for example "Rulebook, Action Phase" or "Reference Guide, page 7").
- When asked whether something is allowed ("can I", "may we", "is it possible"), first look in the passages for a limit or restriction that applies (once per round, only on a City space, not with a Monster present, and so on). If one applies, the answer is no and you state the limit. Say yes only when no passage restricts it.
- Be warm and a little dry; no exclamation marks."""


MAX_PASSAGE_CHARS = 700


def format_passages(passages: list[dict], *, max_chars: int = MAX_PASSAGE_CHARS) -> str:
    """Render retrieved passages as numbered blocks the model can quote from.

    Long chunks are cut at ``max_chars`` (on a word boundary): every prompt token costs
    prefill time before the first spoken word, and the answer needs the gist, not the page.
    """
    blocks = []
    for i, p in enumerate(passages, 1):
        source = p.get("source") or p.get("kind") or "reference"
        where = p.get("path") or p.get("name") or ""
        page = f", page {p['page_start']}" if p.get("page_start") else ""
        text = p["text"]
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[{i}] ({source}: {where}{page})\n{text}")
    return "\n\n".join(blocks) if blocks else "(no passages found)"


CHAT_PROMPT = """You are Reachy, a small desktop robot sitting at a table where people play the board game Eldritch Horror (Fantasy Flight Games, 2013). You are a fellow player with your own investigator; you cannot see the board, so ask people to describe it when it matters.

You are speaking out loud, so answer in one or two short sentences, warm and a little dry, no exclamation marks, no lists. Answer in {language}, keeping game terms in English as printed on the components. If someone asks a rules question, say you will check the Reference Guide rather than guessing."""


def conversation_messages(
    text: str, language: str, recent: list[tuple[str, str]] | None = None
) -> list[dict[str, str]]:
    """Chat messages for table talk that needs no rules lookup (greetings, banter, follow-ups).

    ``recent`` is the last few (speaker_text, robot_text) exchanges for continuity.
    """
    name = language_name(language)
    messages = [{"role": "system", "content": CHAT_PROMPT.format(language=name)}, *history_messages(recent)]
    messages.append({"role": "user", "content": f"{text}\n\nReply in {name}."})
    return messages


MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARS = 300


def history_messages(recent: list[tuple[str, str]] | None) -> list[dict[str, str]]:
    """The last exchanges as chat turns (answers shortened), so follow-ups keep their context."""
    messages: list[dict[str, str]] = []
    for heard, said in (recent or [])[-MAX_HISTORY_TURNS:]:
        said = said if len(said) <= MAX_HISTORY_CHARS else said[:MAX_HISTORY_CHARS].rsplit(" ", 1)[0] + " ..."
        messages.append({"role": "user", "content": heard})
        messages.append({"role": "assistant", "content": said})
    return messages


def rules_question_messages(
    question: str, passages: list[dict], language: str, recent: list[tuple[str, str]] | None = None
) -> list[dict[str, str]]:
    """Chat messages for answering a rules or knowledge question out loud.

    ``recent`` carries the previous exchanges so "and do I need to replace those cards?"
    is understood after a question about acquiring cards.
    """
    name = language_name(language)
    system = SYSTEM_PROMPT.format(language=name)
    # The trailing reminder matters: models follow the last instruction most reliably.
    user = (
        f"Reference passages:\n\n{format_passages(passages)}\n\n"
        f"Question from a player (it may continue the conversation above): {question}\n\n"
        f"Reply in {name}, keeping the game terms in English as printed on the components."
    )
    return [
        {"role": "system", "content": system},
        *history_messages(recent),
        {"role": "user", "content": user},
    ]


DECISION_PROMPT = """You are Reachy, a small desktop robot playing Eldritch Horror (Fantasy Flight Games, 2013, base game only) as a real player at the table. It is your investigator's turn in the Action Phase and you must choose what to do.

You are given the state of the game and a numbered list of turns. Every turn in the list is already legal; your job is which one is best, and why.

How you choose:
- Pick exactly one number from the list. Never invent an action, a space, a card or a rule that is not in what you were given.
- Weigh the game in front of you: how close doom is, where the Gates and Monsters are, how hurt your investigator is, what the active Mystery needs, what the cards you hold and the Reserve cards actually do.
- The rules you may use are the ones written in front of you, and no others. If you do not know what an action costs or tests, say what you are doing without explaining the cost: a wrong rule said out loud is worse than a short sentence.
- Player advice, when it is given to you, is somebody's opinion and is labelled as such. The Rulebook is the authority; advice is not, and you never quote advice as a rule.
- If something you were told is unknown would change your choice, still choose, and put the one question you would ask the table in "ask".

How you speak the reason:
- One to three short sentences, out loud, in {language}. The answer first: what you are doing and the single reason it is better than the alternative. No lists, no markdown, no preamble, no exclamation marks.
- The people at the table cannot see your list. Never mention a number, an option or a shortlist: say what you do, not where it was written.
- Say the action by its printed English name inside your sentence in {language}. In Brazilian Portuguese: "Vou fazer Prepare for Travel e pegar um Train ticket, porque ..." - never "bilhete de trem", never "fase de ação".
- Keep every game term in English exactly as printed: investigator, card and Ancient One names; tokens (Doom, Omen, Clue, Gate, Eldritch, Mystery); phases (Action Phase, Encounter Phase, Mythos Phase); actions (Travel, Rest, Trade, Acquire Assets, Prepare for Travel); conditions (Delayed, Detained); skills (Lore, Influence, Observation, Strength, Will); Health and Sanity.
- Be warm and a little dry, the way a player at the table explains a move."""


DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer"},
        "reason": {"type": "string"},
        "ask": {"type": "string"},
    },
    "required": ["choice", "reason"],
}


def decision_messages(brief: str, options: str, language: str) -> list[dict[str, str]]:
    """Chat messages for choosing one turn out of a shortlist and saying why.

    ``brief`` is the state of the game in words and ``options`` the numbered turns, both built
    by ``src.strategy.decide``; the split keeps this module about wording and that one about
    the game.
    """
    name = language_name(language)
    user = (
        f"{brief}\n\nThe turns you may choose from:\n{options}\n\n"
        f"Answer with JSON only: choice (one number from the list), reason (one to three "
        f"sentences in {name}, spoken out loud), ask (one question for the table, or an empty "
        f"string). Keep the game terms in English as printed."
    )
    return [
        {"role": "system", "content": DECISION_PROMPT.format(language=name)},
        {"role": "user", "content": user},
    ]


PLAN_PROMPT = """You are Reachy, a small desktop robot playing Eldritch Horror (Fantasy Flight Games, 2013, base game only) as a real player. The game has just been set up, or a round has just changed it, and you are deciding how this group intends to win.

A plan is not a turn. It is what you will keep coming back to: which Mysteries you are going for, who covers which part of the map, what your own investigator is for, and what would make you change your mind.

Rules for the plan:
- Use only what you were told. Never invent a card, a space, a Mystery or an investigator that is not in the state you were given, and never state a rule you were not given.
- Say what YOUR investigator is for, in one line, from its own skills and possessions.
- Two to four priorities, each a short line a player would say out loud, in the order you would drop them.
- One to three things to watch for: the events that would make this plan wrong.
- Player advice, when given, is somebody's opinion and is labelled as such; the Rulebook is the authority and advice is not.

Write every line in {language}, keeping the game terms in English exactly as printed: investigator, card and Ancient One names; tokens (Doom, Omen, Clue, Gate, Eldritch, Mystery); phases (Action Phase, Encounter Phase, Mythos Phase); actions (Travel, Rest, Trade, Acquire Assets, Prepare for Travel); skills (Lore, Influence, Observation, Strength, Will); Health and Sanity. In Brazilian Portuguese: "Lily Chen usa Strength e o Protective Amulet para segurar os Monsters" - never "Força", never "Amuleto Protetor", never "Mistérios". Short sentences, warm and a little dry, no lists inside a line, no exclamation marks."""


PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "aim": {"type": "string"},
        "my_role": {"type": "string"},
        "priorities": {"type": "array", "items": {"type": "string"}},
        "watch_for": {"type": "array", "items": {"type": "string"}},
        "changed": {"type": "string"},
    },
    "required": ["aim", "my_role", "priorities", "watch_for"],
}


def plan_messages(brief: str, language: str, previous: str = "", changes: str = "") -> list[dict[str, str]]:
    """Chat messages for writing the plan for this game, or for revising the one in force.

    ``previous`` is the plan as it stands and ``changes`` what happened since it was written;
    with both, the model is asked to keep what still holds and say in "changed" what it dropped.
    """
    name = language_name(language)
    parts = [brief]
    if previous:
        parts.append(f"The plan in force:\n{previous}")
    if changes:
        parts.append(f"What has happened since it was written:\n{changes}")
    parts.append(
        "Answer with JSON only: aim (one sentence), my_role (one line), priorities (two to four "
        "short lines), watch_for (one to three lines)"
        + (", changed (one line on what you dropped from the old plan and why)" if previous else "")
        + f". Every line in {name}, game terms in English as printed."
    )
    return [
        {"role": "system", "content": PLAN_PROMPT.format(language=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


__all__ = [
    "DECISION_PROMPT",
    "PLAN_PROMPT",
    "PLAN_SCHEMA",
    "plan_messages",
    "DECISION_SCHEMA",
    "decision_messages",
    "SYSTEM_PROMPT",
    "CHAT_PROMPT",
    "MAX_PASSAGE_CHARS",
    "conversation_messages",
    "format_passages",
    "history_messages",
    "MAX_HISTORY_TURNS",
    "rules_question_messages",
]
