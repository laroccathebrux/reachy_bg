"""The base-game cards: the 2013 box only, and every effect the printed wording."""

import collections
import json
import re

from src.rag.collections import GAME_ID, KNOWLEDGE_KINDS
from src.rag.ingest_assets import asset, base_game_cards, from_file, main, merge

CARDS = base_game_cards()

ASSET_CATEGORIES = {
    "item",
    "weapon",
    "ally",
    "spell",
    "trinket",
    "service",
    "tome",
    "magical",
    "relic",
    "unique asset",
}
# Arkham Horror content and expansion investigators, which must never appear (hard rule 2).
FOREIGN = re.compile(
    r"\b(Roland Banks|Joe Diamond|elder sign|terror track|Monterey Jack|Agnes Baker|Kate Winthrop)\b",
    re.I,
)


def test_the_card_kinds_are_registered():
    for kind in ("asset", "condition"):
        assert kind in KNOWLEDGE_KINDS


def test_every_card_is_well_formed():
    assert CARDS
    for record in CARDS:
        assert record["kind"] in ("asset", "condition", "task"), record["name"]
        assert record["game_id"] == GAME_ID
        assert record["category"] in ASSET_CATEGORIES | {"condition", "task"}, (
            f"{record['name']}: {record['category']}"
        )
        assert record["base_game"] is True
        assert record["text"].strip()


def test_the_box_holds_what_it_should():
    """76 cards: 65 assets and 11 conditions."""
    kinds = collections.Counter(r["kind"] for r in CARDS)
    assert kinds["asset"] == 65, kinds
    assert kinds["condition"] == 11, kinds
    assert sum(kinds.values()) == 76


def test_every_card_carries_its_printed_effect():
    """The point of the import: something the reasoning layer can act on, not a paraphrase.

    "Bull Whip: +1 Strength during Combat Encounters" can be reasoned about; "helps in a fight"
    cannot, and an invented effect would be worse than either - Influence would be spent on a
    card that does not do what it was told.
    """
    for record in CARDS:
        assert record["confidence"] == "verified", f"{record['name']} has no effect"
        assert len(record["text"]) > 15, f"{record['name']}: {record['text']!r}"
        assert "not been read" not in record["text"]


def test_a_condition_is_not_filed_as_an_asset():
    """A Condition happens *to* an investigator; it is not something they own."""
    conditions = [r for r in CARDS if r["kind"] == "condition"]
    assert conditions
    assert all(r["category"] == "condition" for r in conditions)
    assert not any(r["kind"] == "asset" and r["category"] == "condition" for r in CARDS)


def test_no_expansion_or_arkham_content_leaked_in():
    for record in CARDS:
        assert record["expansion"] == "Eldritch Horror (base game)", record["name"]
        assert not FOREIGN.search(record["text"]), f"{record['name']}: foreign content"


def test_starting_possessions_agree_with_the_investigator_sheets():
    from src.strategy.reference import investigator

    for record in CARDS:
        who = record.get("starting_for", "")
        if not who:
            continue
        sheet = investigator(who)
        assert sheet is not None, f"{record['name']}: unknown investigator {who!r}"
        assert record["name"] in sheet.starting_possessions


def test_every_sheet_possession_has_a_card_with_its_effect():
    """The other direction: what an investigator starts holding must be a card the robot knows."""
    from src.strategy.reference import INVESTIGATORS

    by_name = {r["name"]: r for r in CARDS}
    for sheet in INVESTIGATORS:
        for possession in sheet.starting_possessions:
            record = by_name.get(possession)
            assert record is not None, f"{sheet.name}'s {possession} is missing"
            assert record["confidence"] == "verified", f"{possession} has no effect"
            assert record["starting_for"] == sheet.name


def test_names_are_unique():
    names = [r["name"] for r in CARDS]
    assert len(set(names)) == len(names)


def test_dry_run_writes_nothing(capsys):
    assert main(["--dry-run"]) == 0
    assert "asset" in capsys.readouterr().out


def test_asset_helper_defaults():
    record = asset("X", "item", "A card.")
    assert record["confidence"] == "verified" and record["starting_for"] == ""
    assert record["base_game"] is True


# ---- importing a list that covers the expansions -----------------------------------------


def write_list(tmp_path, items):
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return str(path)


def test_expansion_cards_are_refused(tmp_path):
    """The delivered list held 391 cards; 315 of them were expansion content."""
    records, refused = from_file(
        write_list(
            tmp_path,
            [
                {
                    "name": "Bull Whip",
                    "category": "weapon",
                    "effect": "+1 Strength.",
                    "expansion": "Eldritch Horror",
                },
                {"name": "Ancient Tome", "category": "tome", "effect": "x", "expansion": "Forsaken Lore"},
                {
                    "name": "Dreamlands Thing",
                    "category": "item",
                    "effect": "y",
                    "expansion": "The Dreamlands",
                },
            ],
        )
    )
    assert [r["name"] for r in records] == ["Bull Whip"]
    assert len(refused) == 2 and any("Forsaken Lore" in r for r in refused)


def test_a_card_without_a_set_is_refused(tmp_path):
    """Unverifiable is not the same as base game, which is how expansion content slips in."""
    records, refused = from_file(
        write_list(tmp_path, [{"name": "Dynamite", "category": "weapon", "effect": "boom"}])
    )
    assert records == []
    assert "no expansion" in refused[0]


def test_an_unknown_card_type_is_refused(tmp_path):
    records, refused = from_file(
        write_list(
            tmp_path,
            [{"name": "Weird", "category": "sandwich", "effect": "z", "expansion": "base game"}],
        )
    )
    assert records == []
    assert "unknown card type" in refused[0]


def test_conditions_are_routed_away_from_assets(tmp_path):
    records, _ = from_file(
        write_list(
            tmp_path,
            [
                {"name": "Amnesia", "category": "condition", "effect": "...", "expansion": "Eldritch Horror"},
                {
                    "name": "Axe",
                    "category": "weapon",
                    "effect": "+2 Strength.",
                    "expansion": "Eldritch Horror",
                },
            ],
        )
    )
    by_name = {r["name"]: r for r in records}
    assert by_name["Amnesia"]["kind"] == "condition"
    assert by_name["Axe"]["kind"] == "asset"


def test_merging_never_replaces_a_known_effect_with_a_blank(tmp_path):
    base = base_game_cards()
    known = {r["name"]: r["text"] for r in base}
    incoming, _ = from_file(
        write_list(
            tmp_path,
            [
                {"name": "Kerosene", "category": "item", "expansion": "base game"},  # no effect
                {"name": "Dynamite", "category": "weapon", "effect": "boom", "expansion": "base game"},
            ],
        )
    )
    merged = {r["name"]: r for r in merge(base, incoming)}
    assert merged["Kerosene"]["text"] == known["Kerosene"], "an effectless import overwrote a printed effect"
    assert merged["Dynamite"]["text"] == "boom", "a card the file did not have should come in"


def test_the_shipped_file_is_already_filtered():
    """base_game_cards() raises if the versioned file ever gains an expansion card."""
    from src.rag.ingest_assets import CARDS_FILE

    raw = json.loads(CARDS_FILE.read_text(encoding="utf-8"))
    assert len(raw) == 76
    assert {c["expansion"] for c in raw} == {"Eldritch Horror"}
    assert all(c.get("effect", "").strip() for c in raw)


def test_a_card_is_looked_up_by_name_not_by_embedding():
    """ "What does Bull Whip do?" through the vector search returned Vatican Missionary at 0.32.

    A card's name says almost nothing about its meaning while its effect text dominates the
    embedding, so the name is looked up as the keyword-indexed field it is.
    """
    from src.rag.retrieve import card

    found = card("Bull Whip")
    assert found is not None and found["name"] == "Bull Whip"
    assert "Strength" in found["text"]
    assert card("bull whip")["name"] == "Bull Whip", "lookup must survive how people speak"
    assert card("Amnesia")["kind"] == "condition"
    assert card("Definitely Not A Card") is None


def test_the_reserve_can_be_looked_up_whole():
    """The owner's reserve on 2026-09-16, which is what a buy decision has to weigh."""
    from src.rag.retrieve import cards_in_reserve

    found = cards_in_reserve(["Lucky Cigarette Case", "Private Investigator", "Kerosene", "Bull Whip"])
    assert len(found) == 4
    # "dictated" outranks "verified": somebody read that card off the box (src/rag/dictate.py).
    assert all(c["confidence"] in ("verified", "dictated") for c in found)
