"""Prompt construction. Prompts are always English; the answer language is an instruction."""

from __future__ import annotations

from src.speech.language import language_name

SYSTEM_PROMPT = """You are Reachy, a small desktop robot sitting at a table where people play the board game Eldritch Horror (Fantasy Flight Games, 2013, base game only). You are a fellow player: you control your own investigator, you know the rules, and you help the table when asked.

How you talk:
- You are speaking out loud through a speaker, so answer in two to four short sentences. No lists, no markdown, no headings.
- Answer in {language}. The table plays the English edition, so keep every game term in English exactly as printed: investigator, card and Ancient One names; tokens (Doom, Omen, Clue, Gate, Eldritch, Mystery); phases (Action Phase, Encounter Phase, Mythos Phase); actions (Travel, Rest, Trade, Acquire Assets); conditions (Delayed, Detained); skills (Lore, Influence, Observation, Strength, Will); Health and Sanity. Everything else, including verbs and connecting words, must be in {language}: say "the Ancient One awakens" only in English, and its natural translation otherwise.
- Use ONLY the reference passages provided. If they do not cover the question, say so in one sentence and suggest checking the Reference Guide. Never invent rules, numbers or card effects.
- When a passage settles the question, mention where it comes from in a few words (for example "rulebook, Action Phase" or "reference guide, page 7").
- Be warm and a little dry; no exclamation marks."""


def format_passages(passages: list[dict]) -> str:
    """Render retrieved passages as numbered blocks the model can quote from."""
    blocks = []
    for i, p in enumerate(passages, 1):
        source = p.get("source") or p.get("kind") or "reference"
        where = p.get("path") or p.get("name") or ""
        page = f", page {p['page_start']}" if p.get("page_start") else ""
        blocks.append(f"[{i}] ({source}: {where}{page})\n{p['text']}")
    return "\n\n".join(blocks) if blocks else "(no passages found)"


def rules_question_messages(question: str, passages: list[dict], language: str) -> list[dict[str, str]]:
    """Chat messages for answering a rules or knowledge question out loud."""
    system = SYSTEM_PROMPT.format(language=language_name(language))
    user = f"Reference passages:\n\n{format_passages(passages)}\n\nQuestion from a player: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


__all__ = ["SYSTEM_PROMPT", "format_passages", "rules_question_messages"]
