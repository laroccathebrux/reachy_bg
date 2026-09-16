"""Community strategy advice for Eldritch Horror, in English, into ``bg_knowledge``.

    uv run python -m src.rag.ingest_strategy --dry-run     # print what would be written
    uv run python -m src.rag.ingest_strategy               # write to Qdrant

Strategy is not a rule and must never be answered as one. Every point carries ``kind:
"strategy"``, the source it came from and a ``confidence`` field, so the reasoning layer can
weigh "a player on a forum recommends this" differently from "the rulebook says this". The
rules themselves stay in ``bg_rules``, and entity facts in the other ``bg_knowledge`` kinds.

Sources are recorded per entry rather than per file, so a second guide can be added later
without the robot losing track of who said what.

The first source is a Portuguese community dossier on Ludopedia; the entries below are written
in English (the repository's rule) and are summaries and restatements of the advice, not a
translation of the article. Where this project could only read part of a passage the entry says
so in ``confidence`` and the claim is narrowed to what was actually read.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.logger import get_logger
from src.rag.collections import GAME_ID, KNOWLEDGE
from src.rag.embeddings import embed_texts
from src.rag.store import count, ensure_collection, get_client, point_id, upsert

log = get_logger(__name__)

LUDOPEDIA = {
    "source_kind": "community_guide",
    "source_url": "https://ludopedia.com.br/topico/5625/dossie-eldritch-horror-orientacao-para-jogar",
    "source_author": "yscalybur (Ludopedia forum, 2016)",
    "expansion": "Eldritch Horror (base game)",
    "base_game": True,
}


def entry(
    name: str,
    topic: str,
    text: str,
    *,
    players: int = 0,
    ancient_one: str = "",
    confidence: str = "stated",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "game_id": GAME_ID,
        "kind": "strategy",
        "name": name,
        "topic": topic,
        "players": players,
        "ancient_one": ancient_one,
        "confidence": confidence,
        "text": text,
        **LUDOPEDIA,
        **extra,
    }


# Every entry below is advice from the source, restated in English. None of it is a rule.
ENTRIES: list[dict[str, Any]] = [
    entry(
        "Playing solo with a single investigator",
        "investigator_choice",
        "Advice for one investigator: a single investigator is a hard way to play and the guide "
        "recommends using the easy setup for it, removing the harder Mythos cards. The "
        "disadvantages it lists are concrete. A lone investigator cannot use the cooperative "
        "benefits that trigger when two investigators share a space, and roughly a tenth of the "
        "benefits in the game are of that kind. There is nobody to trade possessions and "
        "resources with. There is nobody to split the map with, so one investigator must travel "
        "to every place a Mystery requires instead of two dividing the board between them. And "
        "penalties that scale with the number of players round up, so a solo investigator takes "
        "the same punishment a two-player game would.",
        players=1,
    ),
    entry(
        "Best investigators for a solo game",
        "investigator_choice",
        "Advice for one investigator: the guide ranks Leo Anderson first, Lily Chen second and "
        "Akachi Onyele third for a solo game.",
        players=1,
        recommended=["Leo Anderson", "Lily Chen", "Akachi Onyele"],
    ),
    entry(
        "Two investigators played together: Jin Culver and Trish Scarborough",
        "investigator_choice",
        "Advice for two investigators kept in the same space: Trish Scarborough leads and Jin "
        "Culver supports her. Trish can always gain a Clue with her ability when she has none, "
        "which makes her efficient at completing tasks that spend Clues, and she can pass spare "
        "Clues to Jin when he needs the dice. Jin can duplicate a reroll. Trish's Sanity is not "
        "high, and Jin can restore it. Trish rolls an extra die in combat and is strong with a "
        ".45 Automatic; Jin's ability helps both Lore and Strength tests and can wound monsters "
        "first. Between them no skill is below 3, so there is always a right investigator for a "
        "given test. The pair is strong against monsters and rarely short of Clues.",
        players=2,
        recommended=["Jin Culver", "Trish Scarborough"],
        formation="together",
    ),
    entry(
        "Two investigators played apart: Jacqueline Fine and Charlie Kane",
        "investigator_choice",
        "Advice for two investigators working in different parts of the map: this pair does not "
        "need to share a space, because neither ability requires a partner beside them. Charlie "
        "Kane buys possessions easily with 4 Influence and hands them to Jacqueline Fine, whose "
        "own Influence is only 1, which lifts her skills high enough to lead both Clue hunting "
        "and monster killing. Charlie also grants extra actions, giving more chances to move or "
        "recover health, can flee monsters as a strategy, and can travel to places that need a "
        "Clue traded in. Jacqueline can send Clues at a distance with her ability, and gains "
        "extra Clues when Charlie receives an uncommon condition.",
        players=2,
        recommended=["Jacqueline Fine", "Charlie Kane"],
        formation="apart",
    ),
    entry(
        "Three investigators: Charlie Kane, Akachi Onyele and Trish Scarborough",
        "investigator_choice",
        "Advice for three investigators: this trio works because each has a distinct job. "
        "Charlie Kane supplies resources that raise Akachi Onyele's Strength and Trish "
        "Scarborough's Lore, and his second ability gives actions to both when they need them. "
        "Akachi Onyele concentrates on closing and sealing gates, which is her speciality. Trish "
        "Scarborough keeps multiplying Clues and killing monsters.",
        players=3,
        recommended=["Charlie Kane", "Akachi Onyele", "Trish Scarborough"],
    ),
    entry(
        "Four or five investigators",
        "investigator_choice",
        "Advice for four or five investigators: the guide suggests combining the smaller groups "
        "it recommends for one, two and three investigators rather than naming a separate team.",
        players=4,
    ),
    entry(
        "Six or more investigators",
        "investigator_choice",
        "Advice for six or more investigators: keep combining the recommended smaller groups, but "
        "from this size on deliberately favour investigators whose abilities and starting "
        "possessions are cooperative, because with that many players someone will almost always "
        "be in a space where they can be helped.",
        players=6,
    ),
    entry(
        "Choosing investigators against the Ancient One",
        "ancient_one_matchup",
        "Advice on matchups: which investigators are at the table matters as much as how they "
        "combine, because each Ancient One except Azathoth has a Reckoning effect that some "
        "investigators trigger badly. For some Ancient Ones the advice is to include a certain "
        "investigator, and for others to leave one out.",
        topic_note="general",
    ),
    entry(
        "Yog-Sothoth: be careful taking Akachi Onyele",
        "ancient_one_matchup",
        "Advice against Yog-Sothoth: be careful about bringing Akachi Onyele. She is very "
        "efficient at sealing gates, which means she will usually be standing on a gate, and "
        "that is exactly what Yog-Sothoth's Reckoning punishes: it advances doom by one.",
        ancient_one="Yog-Sothoth",
        caution=["Akachi Onyele"],
    ),
    entry(
        "Monster statistics and what they imply",
        "monsters",
        "Advice on monsters, from the guide's count of the base-game monster tokens. Their "
        "toughness clusters: about 15% have 1 toughness, 35% have 2, 30% have 3, 15% have 4 and "
        "7% have 5, so more than half sit at 2 or 3. The practical consequence the guide draws is "
        "to always carry a weapon or other supporting asset worth at least +2 dice, so a monster "
        "can be killed in a single combat rather than fought twice. Monster Strength is about 21% "
        "at 1, 50% at 2 and 29% at 3; monster Will is about 4% at 1, 35% at 2, 46% at 3 and 15% "
        "at 4. Roughly half of all monsters (49%) have a Reckoning effect, which is the guide's "
        "main argument against fleeing them: leaving one on the board can cost a great deal, such "
        "as being cursed at every Mythos Phase where its Reckoning fires. Epic monsters are "
        "treated separately: always strong and tough, and they cannot be defeated by effects.",
    ),
    entry(
        "Assets are not optional",
        "assets",
        "Advice on assets: the guide argues a game cannot realistically be won without a minimum "
        "set of assets, and warns against a common habit - buying one weapon early and then "
        "ignoring asset purchases for the rest of the game. Its explanation is that players stop "
        "buying because they have no sense of what an asset is likely to be, how likely it is to "
        "appear, what it is for or whom it suits.",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the records, write nothing")
    args = parser.parse_args(argv)

    for record in ENTRIES:
        if record["kind"] != "strategy" or not record["text"].strip():
            raise ValueError(f"bad record: {record.get('name')}")

    topics: dict[str, int] = {}
    for r in ENTRIES:
        topics[r["topic"]] = topics.get(r["topic"], 0) + 1
    log.info("%d strategy records: %s", len(ENTRIES), topics)

    if args.dry_run:
        for r in ENTRIES:
            print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0

    client = get_client()
    ids = [point_id(GAME_ID, r["kind"], r["source_url"], r["name"]) for r in ENTRIES]
    if len(set(ids)) != len(ids):
        raise ValueError("two strategy entries share a name in the same source")
    vectors = embed_texts(r["text"] for r in ENTRIES)
    ensure_collection(client, KNOWLEDGE)
    written = upsert(client, KNOWLEDGE, ids, vectors, ENTRIES)
    log.info("wrote %d strategy points to %s (total now %d)", written, KNOWLEDGE, count(client, KNOWLEDGE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
