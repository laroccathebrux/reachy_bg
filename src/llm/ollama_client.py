"""Chat with the local reasoning model through Ollama."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import ollama

from src.config import OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    seconds: float
    prompt_tokens: int
    output_tokens: int

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.seconds if self.seconds else 0.0


class LLMError(RuntimeError):
    """Raised when Ollama cannot answer."""


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    num_ctx: int = 8192,
    temperature: float = 0.3,
    keep_alive: str = OLLAMA_KEEP_ALIVE,
    think: bool = False,
    format: Any = None,
) -> Reply:
    """Send a chat and return the assistant reply with timing and token counts.

    ``think=False`` disables the model's reasoning trace (Qwen 3.x) to keep spoken answers
    fast; turn it on for strategy decisions where a longer deliberation is worth the wait.

    ``format`` constrains the output: "json" for any object, or a JSON Schema dict for a
    specific shape. Used where the answer is parsed rather than spoken, so that a model in a
    chatty mood cannot wrap the object in prose.
    """
    client = ollama.Client(host=base_url)
    started = time.perf_counter()
    try:
        response = client.chat(
            model=model,
            messages=messages,
            options={"num_ctx": num_ctx, "temperature": temperature},
            keep_alive=keep_alive,
            think=think,
            **({"format": format} if format is not None else {}),
        )
    except Exception as exc:  # ollama raises its own ResponseError / connection errors
        raise LLMError(f"Ollama chat failed ({model} at {base_url}): {exc}") from exc
    seconds = time.perf_counter() - started
    text = (response.message.content or "").strip()
    return Reply(
        text=text,
        model=model,
        seconds=seconds,
        prompt_tokens=int(getattr(response, "prompt_eval_count", 0) or 0),
        output_tokens=int(getattr(response, "eval_count", 0) or 0),
    )


class StreamStats:
    """Filled in while :func:`chat_stream` runs; read it after the generator is exhausted."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.first_token_s: float | None = None
        self.seconds = 0.0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.text = ""

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.seconds if self.seconds else 0.0


def chat_stream(
    messages: list[dict[str, str]],
    *,
    stats: StreamStats | None = None,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    num_ctx: int = 8192,
    temperature: float = 0.3,
    keep_alive: str = OLLAMA_KEEP_ALIVE,
    think: bool = False,
) -> Iterator[str]:
    """Yield the reply as it is generated (text deltas), so speech can start on the first sentence."""
    client = ollama.Client(host=base_url)
    stats = stats or StreamStats()
    stats.started = time.perf_counter()
    try:
        for chunk in client.chat(
            model=model,
            messages=messages,
            options={"num_ctx": num_ctx, "temperature": temperature},
            keep_alive=keep_alive,
            think=think,
            stream=True,
        ):
            delta = chunk.message.content or ""
            if delta:
                if stats.first_token_s is None:
                    stats.first_token_s = time.perf_counter() - stats.started
                stats.text += delta
                yield delta
            if getattr(chunk, "done", False):
                stats.prompt_tokens = int(getattr(chunk, "prompt_eval_count", 0) or 0)
                stats.output_tokens = int(getattr(chunk, "eval_count", 0) or 0)
    except Exception as exc:
        raise LLMError(f"Ollama stream failed ({model} at {base_url}): {exc}") from exc
    finally:
        stats.seconds = time.perf_counter() - stats.started


__all__ = ["chat", "chat_stream", "Reply", "StreamStats", "LLMError"]
