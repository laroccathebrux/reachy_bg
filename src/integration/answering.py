"""The thinking half of a turn, shared by the typed smoke test and the listening loop.

thought = think("Quantas ações por rodada?", "pt-BR")   # retrieval + LLM
clip = voice(thought)                                   # native-voice TTS
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.llm.ollama_client import chat
from src.llm.prompts import rules_question_messages
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


def voice(thought: Thought) -> Clip:
    """Synthesize the answer with the native voice of its language; records ``tts_s`` / ``audio_s``."""
    t0 = time.perf_counter()
    clip = synthesize(thought.answer, thought.language)
    thought.timings["tts_s"] = round(time.perf_counter() - t0, 2)
    thought.timings["audio_s"] = round(clip.duration_s, 2)
    return clip


__all__ = ["Thought", "think", "voice"]
