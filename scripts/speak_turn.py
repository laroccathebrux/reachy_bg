"""Decide a turn and say it on the robot speaker, without the ElevenLabs agent.

    uv run python scripts/speak_turn.py --demo
    uv run python scripts/speak_turn.py --state data/game.json --lang en-US
    uv run python scripts/speak_turn.py --demo --text "é a vez da Lily Chen"

This is the whole spoken turn minus the microphone and minus the cloud conversation: the local
model decides, the local TTS gives it the native voice, and the audio goes out of the USB device
"Reachy Mini Audio" the same way ``talk.py`` sends the agent's voice. An agent session costs
minutes of the owner's account; this costs a few hundred characters of TTS, so a turn can be
rehearsed as often as it needs to be before anybody spends a session on it.

``--text`` feeds a sentence through the addressee rules first, exactly as the gate does with a
transcript, to check that it is heard as a turn call at all.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AUDIO_OUTPUT_DEVICE, DEFAULT_LANGUAGE  # noqa: E402
from src.integration.turn_taking import TurnTaker  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.speech.agent_audio import RobotAudioInterface  # noqa: E402
from src.speech.tts import TTSError, synthesize  # noqa: E402
from src.strategy.game import GameState  # noqa: E402
from src.strategy.turn import demo_game  # noqa: E402

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", help="a game saved with GameState.save")
    parser.add_argument("--demo", action="store_true", help="the demo setup of src.strategy.turn")
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--text", default="", help="the transcript to judge, instead of taking the turn outright"
    )
    parser.add_argument("--plan", action="store_true", help="write the plan for the game first")
    parser.add_argument("--silent", action="store_true", help="decide and print, but do not speak")
    parser.add_argument("--output-device", default=AUDIO_OUTPUT_DEVICE)
    args = parser.parse_args(argv)

    if args.state:
        game = GameState.load(Path(args.state))
    elif args.demo:
        game = demo_game()
    else:
        parser.error("give --state or --demo")
        return 2

    plan = None
    if args.plan:
        from src.strategy import plan as planner

        plan = planner.make_plan(game, language=args.lang, advice=planner.advice_for(game))
        print(plan.as_text() or "(no plan)")

    audio = None
    if not args.silent:
        audio = RobotAudioInterface(None, args.output_device)  # speak only: no microphone here
        audio.start(lambda _b: None)
        log.info("speaker: %s", audio.output_name)

    def say(text: str, language: str) -> None:
        print(f"\n== says ({language}): {text}")
        if audio is None:
            return
        started = time.perf_counter()
        try:
            clip = synthesize(text, language)
        except TTSError as exc:
            log.error("%s", exc)
            return
        with wave.open(str(clip.path), "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
        print(f"   {clip.duration_s:.1f}s of audio in {time.perf_counter() - started:.1f}s -> speaker")
        audio.output(pcm)
        time.sleep(clip.duration_s + 0.7)  # let the playback thread finish before the stream closes

    taker = TurnTaker(game, say=say, language=args.lang, plan=plan, background=False)
    print(f"== playing {taker.investigator or 'nobody'}: {game.briefing()}")
    try:
        if args.text:
            heard = taker.handle(args.text, args.lang)
            print(f"== {args.text!r} -> {'my turn' if heard else 'not a turn call for me'}")
            if not heard:
                return 1
        else:
            taker._take(args.lang)
        if taker.last is not None:
            print(f"== plan: {taker.last.plan.describe() if taker.last.plan else 'no action'}")
            print(f"== chosen by the {taker.last.chosen_by} in {taker.last.seconds:.1f}s")
    finally:
        if audio is not None:
            audio.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
