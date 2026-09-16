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
