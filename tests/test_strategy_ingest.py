"""The community strategy entries: English only, sourced, and never dressed up as rules."""

import re

import pytest

from src.rag.collections import GAME_ID, KNOWLEDGE_KINDS
from src.rag.ingest_strategy import ENTRIES, entry, main

REQUIRED = {
    "game_id",
    "kind",
    "name",
    "topic",
    "text",
    "players",
    "ancient_one",
    "confidence",
    "expansion",
    "base_game",
    "source_kind",
    "source_url",
    "source_author",
}

# Words that would mean the Portuguese source leaked through untranslated.
PORTUGUESE = re.compile(
    r"\b(não|nao|peça|investigador|investigadores|jogador|jogadores|monstros|recursos"
    r"|ancião|anciao|você|perdição|feitiços|habilidade|dados|combate)\b",
    re.IGNORECASE,
)

# Arkham Horror content, which the project must never mix in (CLAUDE.md, hard rule 2).
ARKHAM = re.compile(r"\b(Roland Banks|Joe Diamond|elder sign|terror track|Miskatonic|Innsmouth)\b", re.I)


def test_strategy_is_a_known_knowledge_kind():
    assert "strategy" in KNOWLEDGE_KINDS


def test_every_entry_has_the_required_fields():
    assert ENTRIES, "no strategy entries"
    for record in ENTRIES:
        missing = REQUIRED - set(record)
        assert not missing, f"{record.get('name')}: missing {sorted(missing)}"
        assert record["kind"] == "strategy"
        assert record["game_id"] == GAME_ID
        assert record["text"].strip()
        assert record["confidence"] in ("stated", "partial")


def test_entries_are_in_english():
    for record in ENTRIES:
        found = PORTUGUESE.findall(record["text"]) + PORTUGUESE.findall(record["name"])
        assert not found, f"{record['name']}: Portuguese left in: {sorted(set(found))}"


def test_no_arkham_horror_content_leaked_in():
    for record in ENTRIES:
        assert not ARKHAM.search(record["text"]), f"{record['name']}: Arkham Horror content"


def test_every_entry_carries_its_source():
    for record in ENTRIES:
        assert record["source_url"].startswith("http")
        assert record["source_author"]
        assert record["source_kind"] == "community_guide"


def test_advice_is_framed_as_advice_not_as_a_rule():
    """The reasoning layer must be able to tell a forum opinion from the rulebook."""
    for record in ENTRIES:
        text = record["text"].lower()
        assert "advice" in text or "the guide" in text, (
            f"{record['name']}: reads like a rule rather than sourced advice"
        )


def test_names_are_unique_within_the_source():
    names = [r["name"] for r in ENTRIES]
    assert len(set(names)) == len(names)


def test_entry_helper_fills_the_defaults():
    record = entry("X", "monsters", "Advice on things.")
    assert record["kind"] == "strategy" and record["players"] == 0
    assert record["ancient_one"] == "" and record["confidence"] == "stated"
    assert record["game_id"] == GAME_ID and record["base_game"] is True


def test_dry_run_writes_nothing(capsys):
    assert main(["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "strategy" in printed and "source_url" in printed


def test_player_counts_cover_the_table_sizes():
    counts = {r["players"] for r in ENTRIES if r["topic"] == "investigator_choice"}
    assert {1, 2, 3} <= counts, f"missing advice for a common table size: {sorted(counts)}"


@pytest.mark.parametrize("record", ENTRIES, ids=lambda r: r["name"][:40])
def test_each_entry_is_long_enough_to_be_useful(record):
    assert len(record["text"]) >= 120, f"{record['name']} is too thin to retrieve well"
