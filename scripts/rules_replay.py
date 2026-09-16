"""Replay the addressee rules over a logged session, to see what a rule change would do.

    uv run python scripts/rules_replay.py --since 17:00
    uv run python scripts/rules_replay.py --log data/game_logs/addressee.jsonl --all

``scripts/gate_replay.py`` replays *audio* through the whole gate (Whisper included), which is
the right test for a transcription or voiceprint change and costs a second per utterance.
This one replays only the decision, from the transcripts already in
``data/game_logs/addressee.jsonl``: it is instant, it covers every utterance of every session
ever logged, and it says exactly which lines a rule change would have flipped.

It is the answer to "the robot ignored what I told it": take the session, change the rule, and
count the lines that change verdict, with their text next to them.

What it cannot replay is the self-echo check, which needs the timing of the robot's own
playback: a line the live gate called ``self_echo`` is judged here on its text alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GAME_LOG_DIR  # noqa: E402
from src.speech.addressee import decide  # noqa: E402

ADDRESSED_ROUTES = {"released", "early", "switched", "my_turn"}


def replay(rows: list[dict], *, humans: int | None = 2, investigator: str = "") -> list[dict]:
    """Judge every logged utterance again, tracking who held the floor as the session went."""
    out: list[dict] = []
    addressed_at: float | None = None
    addressed_by = ""
    addressed_label = ""
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        speaker, label, ts = row.get("speaker") or "", row.get("diart_label") or "", row["ts"]
        if speaker or addressed_by:
            same = bool(speaker) and speaker == addressed_by
        elif label or addressed_label:
            same = bool(label) and label == addressed_label
        else:
            same = False
        since_addressed = (ts - addressed_at) if (same and addressed_at is not None) else None
        verdict = decide(
            text,
            row.get("language") or "pt-BR",
            seconds_since_robot_spoke=row.get("seconds_since_robot_spoke"),
            humans_present=humans,
            other_names=[],
            my_investigator=investigator,
            seconds_since_addressed=since_addressed,
        )
        if verdict.addressed:
            addressed_at, addressed_by, addressed_label = ts, speaker, label
        out.append({**row, "new_addressed": verdict.addressed, "new_reason": verdict.reason})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=str(Path(GAME_LOG_DIR) / "addressee.jsonl"))
    parser.add_argument("--since", default="", help="HH:MM today, or empty for the last session")
    parser.add_argument("--all", action="store_true", help="every line, not only the ones that changed")
    parser.add_argument("--humans", type=int, default=2)
    parser.add_argument("--investigator", default="Lily Chen")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    if args.since:
        hour, _, minute = args.since.partition(":")
        day = time.localtime()
        start = time.mktime((day.tm_year, day.tm_mon, day.tm_mday, int(hour), int(minute or 0), 0, 0, 0, -1))
        rows = [r for r in rows if r.get("ts", 0) >= start]
    elif rows:  # the last session: everything after the longest gap in the file
        gaps = [i for i in range(1, len(rows)) if rows[i]["ts"] - rows[i - 1]["ts"] > 600]
        rows = rows[gaps[-1] :] if gaps else rows

    judged = replay(rows, humans=args.humans, investigator=args.investigator)
    changed = [r for r in judged if r["new_addressed"] != (r["route"] in ADDRESSED_ROUTES)]
    for row in judged:
        was = row["route"] in ADDRESSED_ROUTES
        now = row["new_addressed"]
        if not args.all and was == now:
            continue
        mark = "  " if was == now else ("+ " if now else "- ")
        stamp = time.strftime("%H:%M:%S", time.localtime(row["ts"]))
        print(f"{mark}{stamp} {row['reason']:18s} -> {row['new_reason']:18s} {(row.get('text') or '')[:80]}")
    heard = sum(1 for r in judged if r["route"] in ADDRESSED_ROUTES)
    print(
        f"\n{len(judged)} utterances: {heard} reached the robot, "
        f"{sum(1 for r in judged if r['new_addressed'])} would now "
        f"({len(changed)} changed, {sum(1 for r in changed if r['new_addressed'])} newly heard)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
