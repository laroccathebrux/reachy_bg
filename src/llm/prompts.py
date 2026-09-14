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


__all__ = [
    "SYSTEM_PROMPT",
    "CHAT_PROMPT",
    "MAX_PASSAGE_CHARS",
    "conversation_messages",
    "format_passages",
    "history_messages",
    "MAX_HISTORY_TURNS",
    "rules_question_messages",
]
