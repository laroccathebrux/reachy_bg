"""Play one turn on the screen: the legal options, the shortlist, the choice and the reason.

    uv run python -m src.strategy.turn --demo                    # the owner's two-player setup
    uv run python -m src.strategy.turn --state data/game.json    # a game saved by GameState.save
    uv run python -m src.strategy.turn --demo --advice --lang pt-BR

Text only, on purpose: no microphone, no ElevenLabs, no minutes spent. It is the same call the
spoken turn will make (``src.strategy.decide.decide``), printed instead of said, so the decision
can be argued with before it costs anything to hear.

``--demo`` builds the setup of the live session of 2026-09-16 - Azathoth, Lily Chen with the
robot at Shanghai, Jacqueline Fine with the owner, four cards in the Reserve, a Gate and a
Monster at Rome - because a decision is only worth reading against a board somebody recognises.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.logger import get_logger
from src.strategy import decide as decider
from src.strategy import map_graph, moves
from src.strategy import plan as planner
from src.strategy.game import ROBOT, GameState

log = get_logger(__name__)


class _Sighting:
    """What a scan hands to BoardState, built here by hand for the demo."""

    def __init__(self, space: str, piece_id: int) -> None:
        self.space = space
        self.near = None
        self.x = 0.0
        self.y = 0.0
        self.kind = "piece"
        self.id = piece_id


def demo_game() -> GameState:
    """The board of the last live session, as the robot understood it."""
    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Jacqueline Fine", controller="Alessandro")
    game.mystery = "not told to me yet"
    game.round = 1
    game.phase = "action"
    game.reserve = ["Bull Whip", "Arcane Scholar", "Dynamite", "Old Journal"]
    game.board.seed([_Sighting("Rome", 1), _Sighting("Rome", 2), _Sighting("Shanghai", 3)])
    game.board.name(1, "gate:Rome")
    game.board.name(2, "monster:Cultist")
    game.claim_pieces()
    return game


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", help="a game saved with GameState.save")
    parser.add_argument("--demo", action="store_true", help="use the last live session's setup")
    parser.add_argument("--investigator", default="", help="whose turn; default is the robot's own")
    parser.add_argument("--lang", default="pt-BR", help="language of the spoken reason")
    parser.add_argument("--advice", action="store_true", help="add community strategy entries (needs Qdrant)")
    parser.add_argument("--all", action="store_true", help="print every legal turn, not just the shortlist")
    parser.add_argument("--no-llm", action="store_true", help="score only, do not ask the model")
    parser.add_argument(
        "--plan", action="store_true", help="write the plan for the game first, and follow it"
    )
    parser.add_argument("--shortlist", type=int, default=decider.SHORTLIST)
    args = parser.parse_args(argv)

    if args.state:
        game = GameState.load(Path(args.state))
    elif args.demo:
        game = demo_game()
    else:
        parser.error("give --state or --demo")
        return 2

    state = game.by_name(args.investigator) if args.investigator else game.robot_investigator
    if state is None:
        print(f"I do not know who {args.investigator or 'I'} am playing. {game.briefing()}")
        return 1
    situation = moves.situation_of(game, state)

    print(f"== {game.briefing()}")
    print(f"== {state.describe()}")
    if not map_graph.VERIFIED:
        print("== the path table has not been checked against the board: no turn that moves will be chosen")

    options = moves.candidates(game, situation)
    plans = moves.plans(game, situation)
    print(f"\n{len(options)} actions, {len(plans)} legal turns")
    for candidate in options:
        line = f"  - {candidate.describe(situation.space)}"
        if candidate.unknowns:
            line += f"\n      unknown: {'; '.join(candidate.unknowns)}"
        print(line)
    for refusal in moves.refusals(game, situation):
        print(f"  x {refusal}")

    scored = sorted((decider.score(game, situation, p) for p in plans), key=lambda s: -s.score)
    shown = scored if args.all else decider.shortlist(scored, args.shortlist)
    print(f"\nshortlist ({len(shown)} of {len(scored)}):")
    for item in shown:
        why = f"  <- {'; '.join(item.reasons)}" if item.reasons else ""
        print(f"  {item.score:+5.1f}  {item.plan.describe()}{why}")

    advice = []
    if args.advice:
        from src.rag.retrieve import advice as fetch_advice

        question = (
            f"{game.ancient_one.name if game.ancient_one else ''} turn for {state.name} at {state.space}"
        )
        started = time.perf_counter()
        advice = fetch_advice(question)
        print(f"\n{len(advice)} strategy entries in {time.perf_counter() - started:.1f}s")

    if args.no_llm:
        return 0

    game_plan = None
    if args.plan:
        print("\nwriting the plan for this game ...")
        game_plan = planner.make_plan(
            game, language=args.lang, advice=planner.advice_for(game) if args.advice else None
        )
        print(game_plan.as_text() or "(no plan: the model did not answer)")

    print("\nasking the model ...")
    decision = decider.decide(
        game, state, language=args.lang, limit=args.shortlist, advice=advice or None, plan=game_plan
    )
    print(f"\n== {decision.plan.describe() if decision.plan else 'no action'}")
    print(f"== {decision.reason}")
    if decision.questions:
        print("== would ask: " + "; ".join(decision.questions))
    print(f"== chosen by the {decision.chosen_by} in {decision.seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
