"""End-to-end smoke test: typed question -> retrieval -> LLM -> native-voice TTS -> robot.

    uv run python -m src.integration.smoke --no-robot                   # audio on the Mac speaker
    uv run python -m src.integration.smoke                              # daemon on :8000 required
    uv run python -m src.integration.smoke -q "How many actions per round?" -q "Quantas ações por rodada?"

Every stage is timed so the first real numbers for this Mac (embedding, retrieval, LLM
tokens per second, TTS, playback) come out of the same run.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config import OLLAMA_MODEL, validate_config
from src.llm.ollama_client import LLMError, chat
from src.llm.prompts import rules_question_messages
from src.logger import get_logger
from src.rag.retrieve import retrieve
from src.robot.reachy import Robot
from src.speech.language import detect_language
from src.speech.tts import TTSError, synthesize

log = get_logger(__name__)


def answer(question: str, robot: Robot, *, speak: bool = True) -> dict[str, float | str]:
    """Run one question through the whole chain; return timings and the answer text."""
    timings: dict[str, float | str] = {"question": question}
    language = detect_language(question)
    timings["language"] = language
    log.info("question (%s): %s", language, question)
    robot.antennas(20, -20)

    t0 = time.perf_counter()
    passages = retrieve(question)
    timings["retrieve_s"] = round(time.perf_counter() - t0, 2)
    timings["passages"] = len(passages)
    for p in passages[:3]:
        log.info(
            "  %.3f %s :: %s", p["score"], p.get("path") or p.get("name"), p["text"][:80].replace("\n", " ")
        )

    messages = rules_question_messages(question, passages, language)
    reply = chat(messages)
    timings["llm_s"] = round(reply.seconds, 2)
    timings["llm_tokens_per_s"] = round(reply.tokens_per_second, 1)
    timings["answer"] = reply.text
    log.info("answer (%s): %s", language, reply.text)

    if speak:
        t0 = time.perf_counter()
        clip = synthesize(reply.text, language)
        timings["tts_s"] = round(time.perf_counter() - t0, 2)
        timings["audio_s"] = round(clip.duration_s, 2)
        robot.nod()
        robot.say(clip)
    robot.antennas(0, 0)
    return timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-robot", action="store_true", help="no daemon: log gestures, play audio on the Mac"
    )
    parser.add_argument("--no-speech", action="store_true", help="skip TTS and playback (text only)")
    parser.add_argument(
        "-q", "--question", action="append", help="question(s) to ask; omit for an interactive loop"
    )
    args = parser.parse_args(argv)

    problems = validate_config()
    for problem in problems:
        log.warning("config: %s", problem)
    if problems and not args.no_speech:
        log.error("fix the configuration above or pass --no-speech")
        return 1
    log.info("model %s", OLLAMA_MODEL)

    questions = args.question or []
    with Robot.connect(simulated=args.no_robot) as robot:
        robot.nod()
        if questions:
            for q in questions:
                _run(q, robot, speak=not args.no_speech)
        else:
            print("Type a question in Portuguese or English (empty line to quit).")
            while True:
                try:
                    q = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    break
                _run(q, robot, speak=not args.no_speech)
    return 0


def _run(question: str, robot: Robot, *, speak: bool) -> None:
    try:
        timings = answer(question, robot, speak=speak)
    except (LLMError, TTSError) as exc:
        log.error("%s", exc)
        return
    summary = {k: v for k, v in timings.items() if k not in ("question", "answer")}
    log.info("timings %s", summary)


if __name__ == "__main__":
    sys.exit(main())
