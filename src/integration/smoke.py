"""End-to-end smoke test: typed question -> retrieval -> LLM -> native-voice TTS -> robot.

    uv run python -m src.integration.smoke --no-robot                   # audio on the Mac speaker
    uv run python -m src.integration.smoke                              # daemon on :8000 required
    uv run python -m src.integration.smoke -q "How many actions per round?" -q "Quantas ações por rodada?"

Every stage is timed so the first real numbers for this Mac (embedding, retrieval, LLM
tokens per second, TTS, playback) come out of the same run.
"""

from __future__ import annotations

import argparse
import random
import sys

from src.config import OLLAMA_MODEL, validate_config
from src.integration.answering import think, voice
from src.llm.ollama_client import LLMError
from src.logger import get_logger
from src.robot.reachy import GREETING_MOVES, THINKING_MOVES, Robot
from src.speech.language import detect_language
from src.speech.tts import TTSError

log = get_logger(__name__)


def answer(question: str, robot: Robot, *, speak: bool = True) -> dict[str, float | str]:
    """Run one question through the whole chain; return timings and the answer text."""
    language = detect_language(question)
    log.info("question (%s): %s", language, question)
    # A recorded "thinking" move runs while retrieval and the LLM work.
    robot.emotion(random.choice(THINKING_MOVES), sound=False, block=False)
    thought = think(question, language)
    if speak:
        robot.say(voice(thought))
    return {"question": question, "language": language, "answer": thought.answer, **thought.timings}


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
        robot.emotion(random.choice(GREETING_MOVES))
        try:
            if questions:
                for q in questions:
                    _run(q, robot, speak=not args.no_speech)
            else:
                print("Type a question in Portuguese or English (empty line to quit).")
                while True:
                    try:
                        q = input("> ").strip()
                    except EOFError:
                        break
                    if not q:
                        break
                    _run(q, robot, speak=not args.no_speech)
        except KeyboardInterrupt:
            log.info("interrupted; stopping speech and returning to neutral")
            robot.stop_speaking()
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
