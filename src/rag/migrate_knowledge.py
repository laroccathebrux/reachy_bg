"""Build ``bg_knowledge`` from the legacy ``conhecimento`` collection.

The legacy collection was produced by an earlier project with Portuguese payload keys and
Portuguese template labels wrapped around English source text ("Investigador: Lily Chen ...
Vida (Health) 6, Sanidade (Sanity) 6 ..."). This module reads every point through the Qdrant
API, rewrites keys and labels in English, parses the investigator sheets into structured
fields, replaces the four Ancient One records with verified English descriptions, re-embeds
the English text with bge-m3 and writes the result to ``bg_knowledge``.

Only data crosses over; no code from the earlier project is used.

    uv run python -m src.rag.migrate_knowledge --dry-run     # show the transformed records
    uv run python -m src.rag.migrate_knowledge               # write bg_knowledge
    uv run python -m src.rag.migrate_knowledge --recreate    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from src.logger import get_logger
from src.rag.collections import GAME_ID, KNOWLEDGE
from src.rag.embeddings import embed_texts
from src.rag.store import count, ensure_collection, get_client, point_id, scroll_all, upsert

log = get_logger(__name__)

LEGACY_COLLECTION = "conhecimento"

KIND_MAP = {
    "investigador": "investigator",
    "anciao": "ancient_one",
    "monstro": "monster",
    "portais": "gates",
    "faq": "faq",
    "cartas": "card_composition",
    "fluxo": "flowchart",
    "posses_iniciais": "starting_possessions",
}

EXPANSION_MAP = {
    "Eldritch Horror (caixa base)": "Eldritch Horror (base game)",
    "Base Game": "Eldritch Horror (base game)",
    "FAQ oficial (todas)": "Official FAQ (all products)",
    "várias": "various",
    "Dreamland, The": "The Dreamlands",
}

CLASS_MAP = {
    "Investigador (Seeker)": "Seeker",
    "Guardião (Guardian)": "Guardian",
    "Místico (Mystic)": "Mystic",
    "Altruísta (Altruist)": "Altruist",
    "Erudito (Erudite)": "Erudite",
    "Explorador (Explorer)": "Explorer",
    "Polímata (Polymath)": "Polymath",
}

# Ordered (pattern, replacement) pairs applied to every text field.
LABEL_REWRITES: tuple[tuple[str, str], ...] = (
    (r"Eldritch Horror \(caixa base\)", "Eldritch Horror (base game)"),
    (r"Dreamland, The", "The Dreamlands"),
    (r"Investigador: ", "Investigator: "),
    (r"Expansão: (.*?) — DISPONÍVEL \(caixa base\)\.", r"Expansion: \1 (owned, base game)."),
    (r"Expansão: (.*?) — NÃO disponível \(expansão\)\.", r"Expansion: \1 (expansion, not owned)."),
    (r"Expansão: Base Game — na caixa base\.", "Expansion: base game (owned)."),
    (r"Expansão: (.*?) — expansão, NÃO disponível\.", r"Expansion: \1 (expansion, not owned)."),
    (r"Expansão: ", "Expansion: "),
    (r"Vida \(Health\) (\d+), Sanidade \(Sanity\) (\d+)\.", r"Health \1, Sanity \2."),
    (
        r"Atributos: Conhecimento \(Lore\) (\d+), Influência \(Influence\) (\d+), Observação \(Observation\) (\d+), "
        r"Força \(Strength\) (\d+), Determinação \(Will\) (\d+)\.",
        r"Skills: Lore \1, Influence \2, Observation \3, Strength \4, Will \5.",
    ),
    (r"Ação especial: ", "Action ability: "),
    (r"Habilidade passiva: ", "Passive ability: "),
    (r"Itens iniciais: ", "Starting possessions: "),
    (r"Local inicial: ", "Starting location: "),
    (r"Papel sugerido: ", "Suggested role: "),
    (r"Classe\(s\): ", "Class(es): "),
    (r"Monstro: ", "Monster: "),
    (r"\(Épico\)", "(Epic)"),
    (r"Pergunta: ", "Q: "),
    (r"Resposta: ", "A: "),
    (
        r"Portais \(Gates\) do tabuleiro principal na caixa base, por cor: ",
        "Gates on the main board (base game), by omen color: ",
    ),
    (r"\(texto parcial das cartas iniciais\)", "(partial text of the starting possession cards)"),
    (r"Ancião: ", "Ancient One: "),
)

PORTUGUESE_RESIDUE = re.compile(
    r"\b(Investigador|Expansão|Vida|Sanidade|Atributos|Conhecimento|Influência|Observação|Determinação|"
    r"Pergunta|Resposta|Monstro|Portais|Ancião|caixa base|disponível|Papel sugerido)\b"
)

# Verified against the FFG Reference Guide and the Eldritch Horror wiki (see docs/GAME_REFERENCE.md).
ANCIENT_ONES: dict[str, dict[str, Any]] = {
    "Azathoth": {
        "title": "The Daemon Sultan",
        "doom": 15,
        "mythos_deck": {"stage_1": [1, 2, 1], "stage_2": [2, 3, 1], "stage_3": [2, 4, 0]},
        "setup": "Place 1 Eldritch token on the green space of the Omen track.",
        "front_effect": "When the Omen advances to the green space, advance Doom by 1 for each Eldritch token on that space.",
        "awakening": "The world is devoured: the investigators lose immediately. There is no Final Mystery.",
    },
    "Cthulhu": {
        "title": "The Madness from the Sea",
        "doom": 12,
        "mythos_deck": {"stage_1": [0, 2, 2], "stage_2": [1, 3, 0], "stage_3": [3, 4, 0]},
        "setup": "Set aside 1 Deep One, 1 Star Spawn and all Cthulhu Special Encounter cards.",
        "front_effect": (
            "An investigator who moves onto a space containing an Eldritch token becomes Delayed and loses 1 Sanity. "
            "Reckoning: each investigator on a Sea space without an Eldritch token places one there."
        ),
        "awakening": (
            "Cthulhu spawns as an Epic Monster on Space 3. Final Mystery: defeat the Epic Monster after 3 Mysteries "
            "are solved. Doom advances become Sanity tokens on the sheet; each Reckoning every investigator loses "
            "Sanity per token; the investigators lose when all are eliminated."
        ),
    },
    "Shub-Niggurath": {
        "title": "The Black Goat of the Woods",
        "doom": 13,
        "mythos_deck": {"stage_1": [1, 2, 1], "stage_2": [3, 2, 1], "stage_3": [2, 4, 0]},
        "setup": "Set aside 2 Ghouls, 2 Goat Spawn and 1 Dark Young.",
        "front_effect": "Reckoning: spawn 1 Monster on a random space; then, if 10 or more Monsters are on the board, advance Doom by 2.",
        "awakening": (
            "Shub-Niggurath spawns as an Epic Monster on The Heart of Africa and all Ghouls, Goat Spawn and Dark Young "
            "move there. Final Mystery: defeat the Epic Monster. Each Doom advance spawns a Monster on her space; "
            "6 or more Monsters there means the investigators lose."
        ),
    },
    "Yog-Sothoth": {
        "title": "The Lurker at the Threshold",
        "doom": 14,
        "mythos_deck": {"stage_1": [0, 2, 1], "stage_2": [2, 3, 1], "stage_3": [3, 4, 0]},
        "setup": "Set aside all Yog-Sothoth Special Encounter cards.",
        "front_effect": "Reckoning: each investigator on a space containing a Gate advances Doom by 1 unless he discards 1 Spell.",
        "awakening": (
            "Final Mystery: after 3 Mysteries are solved, an investigator on a Gate space may resolve 'The Key and the "
            "Gate' Special Encounter; the investigators win when the Eldritch tokens on the sheet equal half the "
            "number of investigators. Doom advances place Gates on the sheet instead; 3 or more Gates there means "
            "the investigators lose."
        ),
    },
}

_INVESTIGATOR_RE = re.compile(
    r"Investigator: (?P<name>.+?) \((?P<title>.+?)\)\. Expansion: (?P<expansion>.+?)\.\s*"
    r"Health (?P<health>\d+), Sanity (?P<sanity>\d+)\. "
    r"Skills: Lore (?P<lore>\d+), Influence (?P<influence>\d+), Observation (?P<observation>\d+), "
    r"Strength (?P<strength>\d+), Will (?P<will>\d+)\.\s*"
    r"Action ability: (?P<action>.+?)\s*Passive ability: (?P<passive>.+?)\s*"
    r"Starting possessions: (?P<possessions>.+?)\. Starting location: (?P<location>.+?)\."
    r"(?: Suggested role: (?P<role>.+?)\.)?(?: Class\(es\): (?P<classes>.+?)\.)?\s*$",
    re.DOTALL,
)


def translate_text(text: str) -> str:
    """Rewrite the Portuguese template labels of a legacy text field into English."""
    for pattern, replacement in LABEL_REWRITES:
        text = re.sub(pattern, replacement, text)
    for legacy, english in CLASS_MAP.items():
        text = text.replace(legacy, english)
    return " ".join(text.split())


def parse_investigator(text: str) -> dict[str, Any] | None:
    """Structured fields from a translated investigator sheet, or None when it does not match."""
    match = _INVESTIGATOR_RE.search(text)
    if not match:
        return None
    g = match.groupdict()
    return {
        "occupation": g["title"].strip(),
        "health": int(g["health"]),
        "sanity": int(g["sanity"]),
        "skills": {
            "lore": int(g["lore"]),
            "influence": int(g["influence"]),
            "observation": int(g["observation"]),
            "strength": int(g["strength"]),
            "will": int(g["will"]),
        },
        "action_ability": g["action"].strip(),
        "passive_ability": g["passive"].strip(),
        "starting_possessions": g["possessions"].strip(),
        "starting_location": g["location"].strip(),
        "suggested_role": (g["role"] or "").strip(),
        "classes": [c.strip() for c in (g["classes"] or "").split(",") if c.strip()],
    }


def ancient_one_record(name: str) -> dict[str, Any]:
    spec = ANCIENT_ONES[name]
    deck = spec["mythos_deck"]
    stages = ", ".join(
        f"Stage {i} = {deck[f'stage_{i}'][0]} green / {deck[f'stage_{i}'][1]} yellow / {deck[f'stage_{i}'][2]} blue"
        for i in (1, 2, 3)
    )
    text = (
        f"Ancient One: {name}, {spec['title']}. Expansion: Eldritch Horror (base game). "
        f"Starting Doom {spec['doom']}. Mysteries to solve: 3. Mythos deck: {stages}. "
        f"Setup: {spec['setup']} Effect: {spec['front_effect']} Awakening: {spec['awakening']}"
    )
    return {
        "game_id": GAME_ID,
        "kind": "ancient_one",
        "name": name,
        "expansion": "Eldritch Horror (base game)",
        "base_game": True,
        "text": text,
        "title": spec["title"],
        "starting_doom": spec["doom"],
        "mysteries_to_solve": 3,
        "mythos_deck": deck,
        "setup": spec["setup"],
        "front_effect": spec["front_effect"],
        "awakening": spec["awakening"],
    }


def transform(legacy: dict[str, Any]) -> dict[str, Any] | None:
    """Legacy payload -> English payload for ``bg_knowledge``; None to drop the record."""
    kind = KIND_MAP.get(str(legacy.get("tipo", "")))
    if kind is None:
        return None
    name = " ".join(str(legacy.get("nome", "")).split())
    if kind == "ancient_one":
        return ancient_one_record(name) if name in ANCIENT_ONES else None
    expansion = str(legacy.get("expansao", ""))
    expansion = EXPANSION_MAP.get(expansion, expansion)
    text = translate_text(str(legacy.get("texto", "")))
    record: dict[str, Any] = {
        "game_id": GAME_ID,
        "kind": kind,
        "name": name,
        "expansion": expansion,
        "base_game": bool(legacy.get("base", False)),
        "text": text,
    }
    if legacy.get("pagina") is not None:
        record["page"] = legacy["pagina"]
    if kind == "investigator":
        parsed = parse_investigator(text)
        if parsed:
            record.update(parsed)
        else:
            log.warning("could not parse investigator sheet for %s", name)
    return record


def find_residue(record: dict[str, Any]) -> list[str]:
    """Portuguese words left in the text (should be empty after translation)."""
    return sorted(set(PORTUGUESE_RESIDUE.findall(record["text"])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="transform and print; do not write")
    parser.add_argument("--recreate", action="store_true", help="drop bg_knowledge before writing")
    args = parser.parse_args(argv)

    client = get_client()
    if not client.collection_exists(LEGACY_COLLECTION):
        log.error("legacy collection %s not found", LEGACY_COLLECTION)
        return 1
    legacy_points = scroll_all(client, LEGACY_COLLECTION)
    log.info("read %d legacy points from %s", len(legacy_points), LEGACY_COLLECTION)

    records: list[dict[str, Any]] = []
    dropped = 0
    for point in legacy_points:
        record = transform(point["payload"])
        if record is None:
            dropped += 1
            continue
        residue = find_residue(record)
        if residue:
            log.warning("%s '%s' still contains Portuguese: %s", record["kind"], record["name"], residue)
        records.append(record)
    kinds: dict[str, int] = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    log.info("transformed %d records (%d dropped): %s", len(records), dropped, kinds)

    if args.dry_run:
        for r in records:
            print(json.dumps(r, ensure_ascii=False)[:400])
        return 0

    ids = [point_id(GAME_ID, r["kind"], r["name"], r.get("page", ""), r["expansion"]) for r in records]
    if len(set(ids)) != len(ids):
        # Same name on several pages/expansions is legitimate; make ids unique by position.
        ids = [
            point_id(GAME_ID, r["kind"], r["name"], r.get("page", ""), r["expansion"], i)
            for i, r in enumerate(records)
        ]
    vectors = embed_texts(r["text"] for r in records)
    ensure_collection(client, KNOWLEDGE, recreate=args.recreate)
    written = upsert(client, KNOWLEDGE, ids, vectors, records)
    log.info("wrote %d points to %s (total now %d)", written, KNOWLEDGE, count(client, KNOWLEDGE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
