"""The asset cards: what is known is recorded, what is not known is not invented."""

import re

from src.rag.collections import GAME_ID, KNOWLEDGE_KINDS
from src.rag.ingest_assets import ENTRIES, asset, main

CATEGORIES = {"item", "weapon", "ally", "spell", "trinket", "service"}
ARKHAM = re.compile(r"\b(Roland Banks|Joe Diamond|elder sign|terror track|Monterey Jack|Agnes Baker)\b", re.I)


def test_asset_is_a_known_knowledge_kind():
    assert "asset" in KNOWLEDGE_KINDS


def test_every_asset_is_well_formed():
    assert ENTRIES
    for record in ENTRIES:
        assert record["kind"] == "asset" and record["game_id"] == GAME_ID
        assert record["category"] in CATEGORIES, f"{record['name']}: {record['category']}"
        assert record["confidence"] in ("verified", "effect_unchecked")
        assert record["base_game"] is True
        assert record["text"].strip()


def test_an_unread_effect_says_so_instead_of_guessing():
    """A plausible wrong effect is worse than a gap: the reasoning layer would act on it."""
    for record in ENTRIES:
        if record["confidence"] == "effect_unchecked":
            assert "not been read" in record["text"], f"{record['name']} claims an effect it has not verified"


def test_no_expansion_content_leaked_in():
    """The existing starting_possessions points are OCR debris naming expansion investigators."""
    for record in ENTRIES:
        assert not ARKHAM.search(record["text"]), f"{record['name']}: expansion content"


def test_starting_possessions_agree_with_the_investigator_sheets():
    """A card said to start with an investigator must actually be on that sheet."""
    from src.strategy.reference import investigator

    for record in ENTRIES:
        who = record.get("starting_for", "")
        if not who:
            continue
        sheet = investigator(who)
        assert sheet is not None, f"{record['name']}: unknown investigator {who!r}"
        assert record["name"] in sheet.starting_possessions, (
            f"{record['name']} is not among {who}'s starting possessions {sheet.starting_possessions}"
        )


def test_every_sheet_possession_has_an_asset_entry():
    """The other direction: no starting possession is left without a card in the knowledge base."""
    from src.strategy.reference import INVESTIGATORS

    known = {r["name"] for r in ENTRIES}
    for sheet in INVESTIGATORS:
        for possession in sheet.starting_possessions:
            assert possession in known, f"{sheet.name}'s {possession} has no asset entry"


def test_names_are_unique():
    names = [r["name"] for r in ENTRIES]
    assert len(set(names)) == len(names)


def test_dry_run_writes_nothing(capsys):
    assert main(["--dry-run"]) == 0
    assert "asset" in capsys.readouterr().out


def test_asset_helper_defaults():
    record = asset("X", "item", "A card.")
    assert record["confidence"] == "verified" and record["starting_for"] == ""
    assert record["base_game"] is True


# ---- importing a checked list ------------------------------------------------------------


def write_list(tmp_path, items):
    import json as _json

    path = tmp_path / "assets.json"
    path.write_text(_json.dumps(items), encoding="utf-8")
    return str(path)


def test_an_expansion_card_is_refused(tmp_path):
    """Nearly every asset list online mixes the eight expansions in; the set is what filters."""
    from src.rag.ingest_assets import from_file

    records, refused = from_file(
        write_list(
            tmp_path,
            [
                {
                    "name": "Bull Whip",
                    "category": "weapon",
                    "effect": "+2 Strength.",
                    "expansion": "Eldritch Horror",
                },
                {"name": "Ancient Tome", "category": "item", "effect": "x", "expansion": "Forsaken Lore"},
                {
                    "name": "Shrivelling Wand",
                    "category": "spell",
                    "effect": "y",
                    "expansion": "Mountains of Madness",
                },
            ],
        )
    )
    assert [r["name"] for r in records] == ["Bull Whip"]
    assert len(refused) == 2
    assert any("Forsaken Lore" in r for r in refused)


def test_a_card_without_a_set_is_refused(tmp_path):
    from src.rag.ingest_assets import from_file

    records, refused = from_file(
        write_list(tmp_path, [{"name": "Dynamite", "category": "weapon", "effect": "boom"}])
    )
    assert records == []
    assert "no expansion" in refused[0]


def test_an_effect_makes_an_entry_verified(tmp_path):
    from src.rag.ingest_assets import from_file

    records, _ = from_file(
        write_list(
            tmp_path,
            [
                {
                    "name": "Bull Whip",
                    "category": "weapon",
                    "effect": "+2 Strength.",
                    "expansion": "base game",
                },
                {"name": "Rope", "category": "item", "expansion": "base game"},
            ],
        )
    )
    by_name = {r["name"]: r for r in records}
    assert by_name["Bull Whip"]["confidence"] == "verified"
    assert by_name["Rope"]["confidence"] == "effect_unchecked"


def test_merging_only_upgrades_entries_that_gain_an_effect(tmp_path):
    """An import must not overwrite a known card with a vaguer version of itself."""
    from src.rag.ingest_assets import ENTRIES, from_file, merge

    incoming, _ = from_file(
        write_list(
            tmp_path,
            [
                {"name": "Kerosene", "category": "item", "expansion": "base game"},  # no effect
                {
                    "name": "Bull Whip",
                    "category": "weapon",
                    "effect": "+2 Strength.",
                    "expansion": "base game",
                },
                {"name": "Dynamite", "category": "weapon", "effect": "boom", "expansion": "base game"},
            ],
        )
    )
    merged = {r["name"]: r for r in merge(ENTRIES, incoming)}
    assert "starting possession of Mark Harrigan" in merged["Kerosene"]["text"], (
        "an effectless import overwrote what was already known"
    )
    assert merged["Bull Whip"]["confidence"] == "verified"
    assert merged["Dynamite"]["text"] == "boom", "a new card should come in"
