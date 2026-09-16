"""Read a card from the box into the knowledge base, one card at a time.

    uv run python scripts/dictate_cards.py --kind mystery --ancient-one Cthulhu
    uv run python scripts/dictate_cards.py --text "Bull Whip, value 2. +1 Strength during Combat Encounters, reroll 1 die."
    uv run python scripts/dictate_cards.py --list

Type (or paste) what the card says; the local model puts it into fields, the reference checks
the names, and the card goes to ``data/cards/dictated.json`` and into ``bg_knowledge``. Nothing
leaves the machine and nothing is guessed: what the robot ends up knowing is what was read to it.

This exists because the rest of the game is not published anywhere the robot may take it from.
The 76 asset cards and the 16 Mysteries are in the repository - the assets with their printed
effects, the Mysteries as what they ask for - and everything else (the encounter decks, the
Mythos cards) is the box's own text. When the table needs the robot to know one exactly, they
read it, once.

Ctrl-D (or an empty line twice) ends the session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_LANGUAGE  # noqa: E402
from src.rag.dictate import CARD_KINDS, CARDS_FILE, ingest, load, read_card, save  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="", help="one card, instead of reading from the terminal")
    parser.add_argument("--kind", default="", choices=("", *CARD_KINDS), help="what kind of card it is")
    parser.add_argument("--ancient-one", default="", help="whose deck it belongs to, for a Mystery")
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE)
    parser.add_argument("--read-by", default="", help="who read it, kept with the card")
    parser.add_argument("--no-ingest", action="store_true", help="write the file, do not touch Qdrant")
    parser.add_argument("--list", action="store_true", help="print what has been dictated so far")
    args = parser.parse_args(argv)

    if args.list:
        cards = load()
        for card in cards:
            print(f"{card.get('kind', '?'):10s} {card.get('name', '?'):34s} {card.get('text', '')[:90]}")
        print(f"\n{len(cards)} cards in {CARDS_FILE}")
        return 0

    lines: list[str] = [args.text] if args.text else []
    if not lines:
        print("Read one card per line (Ctrl-D to stop).")
        for line in sys.stdin:
            if line.strip():
                lines.append(line.strip())
    if not lines:
        print("nothing read")
        return 1

    kept = []
    for said in lines:
        reading = read_card(
            said,
            language=args.lang,
            read_by=args.read_by,
            kind_hint=args.kind,
            ancient_one_hint=args.ancient_one,
        )
        print(f"\n{reading.describe()}  ({reading.seconds:.1f}s)")
        for problem in reading.problems:
            print(f"  ! {problem}")
        if reading.record:
            print(f"  {reading.record['text']}")
            kept.append(reading.record)
    if not kept:
        return 1
    path = save(kept)
    print(f"\n{len(kept)} card(s) -> {path}")
    if not args.no_ingest:
        print(f"{ingest(kept)} card(s) -> bg_knowledge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
