"""Chat with the local reasoning model through Ollama."""

from __future__ import annotations

import time
from dataclasses import dataclass

import ollama

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL


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
    keep_alive: str = "10m",
    think: bool = False,
) -> Reply:
    """Send a chat and return the assistant reply with timing and token counts.

    ``think=False`` disables the model's reasoning trace (Qwen 3.x) to keep spoken answers
    fast; turn it on for strategy decisions where a longer deliberation is worth the wait.
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


__all__ = ["chat", "Reply", "LLMError"]
