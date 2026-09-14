"""The thinking half of a turn, shared by the typed smoke test and the listening loop.

thought = think("Quantas ações por rodada?", "pt-BR")   # retrieval + LLM
clip = voice(thought)                                   # native-voice TTS
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from src.llm.ollama_client import StreamStats, chat, chat_stream
from src.llm.prompts import conversation_messages, rules_question_messages
from src.logger import get_logger
from src.rag.retrieve import retrieve
from src.speech.tts import Clip, synthesize

log = get_logger(__name__)


@dataclass
class Thought:
    question: str
    language: str
    answer: str
    passages: list[dict[str, Any]]
    timings: dict[str, float] = field(default_factory=dict)


def think(question: str, language: str) -> Thought:
    """Retrieve reference passages and ask the LLM for a spoken answer in ``language``."""
    t0 = time.perf_counter()
    passages = retrieve(question)
    retrieve_s = time.perf_counter() - t0
    for p in passages[:3]:
        log.info(
            "  %.3f %s :: %s", p["score"], p.get("path") or p.get("name"), p["text"][:80].replace("\n", " ")
        )
    reply = chat(rules_question_messages(question, passages, language))
    log.info("answer (%s): %s", language, reply.text)
    return Thought(
        question=question,
        language=language,
        answer=reply.text,
        passages=passages,
        timings={
            "retrieve_s": round(retrieve_s, 2),
            "passages": len(passages),
            "llm_s": round(reply.seconds, 2),
            "llm_tokens_per_s": round(reply.tokens_per_second, 1),
        },
    )


SPOKEN_RULES_PASSAGES = 3
SPOKEN_KNOWLEDGE_PASSAGES = 2
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_MIN_SENTENCE_CHARS = 12


def split_sentences(buffer: str, *, final: bool = False) -> tuple[list[str], str]:
    """Complete sentences in ``buffer`` and the remainder (everything when ``final``).

    A sentence is text up to ``.``, ``!``, ``?`` or ``…`` followed by whitespace; very short
    pieces ("Dr." or "1.") are glued to the next one.
    """
    parts = _SENTENCE_END.split(buffer)
    if not final:
        rest = parts.pop() if parts else ""
    else:
        rest = ""
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences and len(sentences[-1]) < _MIN_SENTENCE_CHARS:
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    if final and sentences and len(sentences[-1]) < _MIN_SENTENCE_CHARS and len(sentences) > 1:
        last = sentences.pop()
        sentences[-1] = f"{sentences[-1]} {last}"
    return sentences, rest


def think_sentences(
    question: str,
    language: str,
    thought: Thought,
    *,
    lookup: bool = True,
    recent: list[tuple[str, str]] | None = None,
) -> Iterator[str]:
    """Stream the LLM answer sentence by sentence, filling ``thought`` as it goes.

    ``lookup=True`` retrieves reference passages first (rules questions); ``lookup=False``
    sends a short conversation prompt with the last exchanges (banter, follow-ups), which
    keeps the prompt small and the first sentence fast.
    """
    if lookup:
        t0 = time.perf_counter()
        # Fewer passages than the typed smoke test: prompt processing is the biggest share of
        # the time to the first spoken sentence on a busy Mac (about 5 ms per prompt token).
        passages = retrieve(
            question, rules_limit=SPOKEN_RULES_PASSAGES, knowledge_limit=SPOKEN_KNOWLEDGE_PASSAGES
        )
        thought.passages = passages
        thought.timings["retrieve_s"] = round(time.perf_counter() - t0, 2)
        thought.timings["passages"] = len(passages)
        for p in passages[:3]:
            log.info(
                "  %.3f %s :: %s",
                p["score"],
                p.get("path") or p.get("name"),
                p["text"][:80].replace("\n", " "),
            )
        messages = rules_question_messages(question, passages, language)
    else:
        thought.timings["passages"] = 0
        messages = conversation_messages(question, language, recent)
    stats = StreamStats()
    buffer = ""
    first_sentence_at: float | None = None
    for delta in chat_stream(messages, stats=stats):
        buffer += delta
        sentences, buffer = split_sentences(buffer)
        for sentence in sentences:
            if first_sentence_at is None:
                first_sentence_at = time.perf_counter()
                thought.timings["llm_first_sentence_s"] = round(first_sentence_at - stats.started, 2)
            yield sentence
    sentences, _ = split_sentences(buffer, final=True)
    for sentence in sentences:
        if first_sentence_at is None:
            thought.timings["llm_first_sentence_s"] = round(time.perf_counter() - stats.started, 2)
        yield sentence
    thought.answer = " ".join(stats.text.split())
    thought.timings["llm_prompt_tokens"] = stats.prompt_tokens
    thought.timings["llm_s"] = round(stats.seconds, 2)
    thought.timings["llm_tokens_per_s"] = round(stats.tokens_per_second, 1)
    log.info("answer (%s): %s", language, thought.answer)


def voice(thought: Thought) -> Clip:
    """Synthesize the answer with the native voice of its language; records ``tts_s`` / ``audio_s``."""
    t0 = time.perf_counter()
    clip = synthesize(thought.answer, thought.language)
    thought.timings["tts_s"] = round(time.perf_counter() - t0, 2)
    thought.timings["audio_s"] = round(clip.duration_s, 2)
    return clip


__all__ = ["Thought", "split_sentences", "think", "think_sentences", "voice"]
