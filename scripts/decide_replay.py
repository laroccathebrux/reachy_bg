"""How much does the reasoning model add over the score that already ranks the turns?

``decide()`` scores every legal turn, keeps a shortlist, and asks the model to pick one of it.
When the model does not answer, ``kept[0]`` - the best-scored turn - is the decision, and the
whole pipeline still works. So the model is worth what it adds *over that fallback*, and this
measures it on the same situations, with whatever models are installed:

    uv run python scripts/decide_replay.py
    uv run python scripts/decide_replay.py --models qwen3.6:35b-mlx,qwen2.5:3b --repeat 3

For each situation it records what the score alone would pick, then what each model picks, and
reports how often they agree, how long each took, and what the disagreements actually were. A
model that agrees with the score almost always is being paid for nothing; one that disagrees
has to be judged by the disagreements, which are printed in full.

The situations are built here rather than replayed from a log, because a log only ever holds
the turn that was taken - never the ones the robot was choosing between.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import OLLAMA_MODEL
from src.llm.ollama_client import LLMError, chat
from src.logger import get_logger
from src.strategy import moves
from src.strategy.decide import DEFAULT_WEIGHTS, NUM_CTX, decide, score, shortlist
from src.strategy.game import ROBOT, GameState

log = get_logger(__name__)


def _game(
    *,
    space: str = "Shanghai",
    health: int = 6,
    sanity: int = 6,
    clues: int = 0,
    tickets: tuple[int, int] = (0, 0),
    reserve: tuple[str, ...] = ("Lucky Cigarette Case", "Private Investigator", "Kerosene", "Bull Whip"),
    others: tuple[tuple[str, str], ...] = (("Jacqueline Fine", "Alessandro"),),
    doom: int = 15,
) -> GameState:
    game = GameState()
    game.set_ancient_one("Azathoth")
    game.doom = doom
    game.mystery = "The Deep One's Attack"
    game.reserve = list(reserve)
    mine = game.add_investigator("Lily Chen", controller=ROBOT, space=space)
    assert mine is not None
    mine.health, mine.sanity, mine.clues = health, sanity, clues
    mine.train_tickets, mine.ship_tickets = tickets
    for name, who in others:
        game.add_investigator(name, controller=who, space=space if who == "same" else "")
    game.begin_round()
    return game


SITUATIONS: list[tuple[str, GameState]] = [
    ("fresh start at Shanghai", _game()),
    ("hurt: 2 health, 3 sanity", _game(health=2, sanity=3)),
    ("already holding two Train tickets", _game(tickets=(2, 0))),
    ("an empty Reserve", _game(reserve=())),
    ("three Clues in hand", _game(clues=3)),
    ("doom at 3, the end is near", _game(doom=3)),
    ("another investigator on the same space", _game(others=(("Jacqueline Fine", "same"),))),
    ("hurt and holding tickets", _game(health=2, sanity=2, tickets=(1, 1))),
]


@dataclass
class Run:
    model: str
    picks: list[int] = field(default_factory=list)  # 1-based index into the shortlist
    seconds: list[float] = field(default_factory=list)
    failures: int = 0


def _shortlist_for(game: GameState) -> list[Any]:
    state = game.robot_investigator
    assert state is not None
    situation = moves.situation_of(game, state)
    plans = [p for p in moves.plans(game, situation) if not any(a.key == moves.TRAVEL for a in p.actions)]
    return shortlist([score(game, situation, p, DEFAULT_WEIGHTS) for p in plans])


def _ask(model: str) -> Any:
    def chat_fn(messages: list[dict[str, str]], **kwargs: Any) -> Any:
        kwargs.setdefault("num_ctx", NUM_CTX)
        return chat(messages, model=model, **kwargs)

    return chat_fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=OLLAMA_MODEL, help="comma-separated Ollama models to compare")
    parser.add_argument("--repeat", type=int, default=2, help="runs per situation, to see its spread")
    parser.add_argument("--language", default="en-US")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    runs: dict[str, Run] = {m: Run(m) for m in models}
    baselines: list[str] = []
    disagreements: list[str] = []

    for label, game in SITUATIONS:
        kept = _shortlist_for(game)
        if not kept:
            log.warning("%s: no legal turn, skipping", label)
            continue
        by_score = kept[0].plan.describe()
        baselines.append(f"{label}: {by_score}")
        print(f"\n== {label}  ({len(kept)} on the shortlist)")
        print(f"   score alone -> {by_score}")

        for model in models:
            run = runs[model]
            for _ in range(args.repeat):
                started = time.perf_counter()
                try:
                    decision = decide(game, language=args.language, chat_fn=_ask(model))
                except LLMError as exc:
                    run.failures += 1
                    print(f"   {model}: no answer ({exc})")
                    continue
                run.seconds.append(time.perf_counter() - started)
                if decision.chosen_by != "model" or decision.plan is None:
                    run.failures += 1
                    print(f"   {model}: fell back to the score")
                    continue
                chosen = decision.plan.describe()
                index = next((i for i, k in enumerate(kept, 1) if k.plan.describe() == chosen), 0)
                run.picks.append(index)
                mark = "=" if chosen == by_score else "!"
                print(f"   {model}: [{mark}] {chosen}")
                if chosen != by_score:
                    disagreements.append(
                        f"{label}\n     score: {by_score}\n     {model}: {chosen}\n"
                        f"     said : {decision.reason.strip()[:160]}"
                    )

    print("\n" + "=" * 78)
    print(f"{len(baselines)} situations, {args.repeat} run(s) each")
    for model, run in runs.items():
        total = len(run.picks)
        if not total:
            print(f"\n{model}: no usable answer in {run.failures} attempt(s)")
            continue
        same = sum(1 for p in run.picks if p == 1)
        seconds = run.seconds or [0.0]
        print(
            f"\n{model}"
            f"\n  agreed with the score : {same}/{total} ({100 * same / total:.0f}%)"
            f"\n  seconds per decision  : median {statistics.median(seconds):.1f}, "
            f"worst {max(seconds):.1f}"
            f"\n  fell back / failed    : {run.failures}"
        )
    if disagreements:
        print("\n" + "-" * 78)
        print("Where they disagreed - this is what the model is being paid for:\n")
        for item in disagreements:
            print("  " + item + "\n")
    else:
        print("\nNo disagreement at all: the score alone would have taken every one of these turns.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
