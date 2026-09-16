"""Cards read from the box: checked, stored, and never guessed at."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.rag import dictate


@dataclass
class _Reply:
    text: str
    model: str = "test"
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


def _answers(payload: dict):
    def chat_fn(messages, **kwargs):
        return _Reply(json.dumps(payload))

    return chat_fn


def test_a_dictated_mystery_becomes_a_knowledge_point():
    reading = dictate.read_card(
        "Mystery de Cthulhu, The Deep Ones Attack. Coloque um Eldritch token no mar mais próximo.",
        chat_fn=_answers(
            {
                "name": "The Deep Ones Attack!",
                "kind": "mystery",
                "ancient_one": "Cthulhu",
                "text": "Place an Eldritch token on the nearest Sea space.",
            }
        ),
    )
    assert reading.ok
    assert reading.record["kind"] == "mystery"
    assert reading.record["ancient_one"] == "Cthulhu"
    assert reading.record["confidence"] == "dictated"
    assert reading.record["base_game"] is True
    assert "The Deep Ones Attack!" in reading.record["text"]


def test_an_asset_keeps_the_value_printed_on_it():
    # The number the robot has to ask the table for today.
    reading = dictate.read_card(
        "Bull Whip, value 2.",
        chat_fn=_answers({"name": "Bull Whip", "kind": "asset", "text": "+1 Strength.", "value": 2}),
    )
    assert reading.record["value"] == 2


def test_an_ancient_one_the_box_does_not_have_is_refused_with_candidates():
    reading = dictate.read_card(
        "Mystery de Nyarlathotep",
        chat_fn=_answers(
            {
                "name": "Cult of the Bloody Tongue",
                "kind": "mystery",
                "ancient_one": "Nyarlathotep",
                "text": "x",
            }
        ),
    )
    assert not reading.ok
    assert any("not an Ancient One of the base game" in p for p in reading.problems)


def test_a_mystery_without_its_ancient_one_is_a_problem_not_a_guess():
    reading = dictate.read_card(
        "um mistério aí", chat_fn=_answers({"name": "The True Name", "kind": "mystery", "text": "x"})
    )
    assert any("belongs to an Ancient One" in p for p in reading.problems)


def test_a_kind_that_is_not_a_card_in_this_box_is_refused():
    reading = dictate.read_card(
        "uma coisa", chat_fn=_answers({"name": "Focus", "kind": "focus_token", "text": "x"})
    )
    assert reading.record == {}
    assert "not a kind of card" in reading.problems[0]


def test_nothing_read_is_nothing_stored():
    reading = dictate.read_card("aham", chat_fn=_answers({"name": "", "kind": "mystery", "text": ""}))
    assert reading.record == {}
    assert reading.describe().startswith("I did not catch a card")


def test_reading_the_same_card_again_corrects_it_instead_of_doubling_it(tmp_path):
    path = tmp_path / "dictated.json"
    first = {"kind": "mystery", "name": "The True Name", "text": "old"}
    second = {"kind": "mystery", "name": "The True Name", "text": "new"}
    dictate.save([first], path)
    dictate.save([second], path)
    stored = dictate.load(path)
    assert len(stored) == 1 and stored[0]["text"] == "new"
    assert dictate.point_id(first) == dictate.point_id(second)


def test_the_hints_fill_in_what_the_model_left_out():
    reading = dictate.read_card(
        "The True Name. Eldritch tokens em espaços aleatórios.",
        kind_hint="mystery",
        ancient_one_hint="Azathoth",
        chat_fn=_answers({"name": "The True Name", "kind": "", "text": "Eldritch tokens on random spaces."}),
    )
    assert reading.record["kind"] == "mystery"
    assert reading.record["ancient_one"] == "Azathoth"


def test_a_corrupt_file_reads_as_no_cards(tmp_path):
    path = tmp_path / "dictated.json"
    path.write_text("{not json", encoding="utf-8")
    assert dictate.load(path) == []


def test_an_encounter_is_filed_by_its_number_and_its_city():
    reading = dictate.read_card(
        "Carta 8, a parte de Rome: teste Influence, se passar ganha 2 Clues.",
        chat_fn=_answers(
            {
                "name": "",
                "kind": "encounter",
                "number": 8,
                "space": "Rome",
                "text": "Test Influence. If you pass, gain 2 Clues.",
            }
        ),
    )
    assert reading.record["name"] == "Europe 8, Rome"  # the deck follows from the city
    assert reading.record["deck"] == "Europe"
    assert reading.record["number"] == 8 and reading.record["space"] == "Rome"


def test_an_encounter_without_its_number_says_so():
    reading = dictate.read_card(
        "um encontro aí em Rome",
        chat_fn=_answers({"name": "Something", "kind": "encounter", "space": "Rome", "text": "x"}),
    )
    assert any("number on it" in p for p in reading.problems)


def test_a_city_that_is_not_on_the_board_is_refused():
    reading = dictate.read_card(
        "carta 8, parte de Gotham",
        chat_fn=_answers({"name": "x", "kind": "encounter", "number": 8, "space": "Gotham", "text": "x"}),
    )
    assert any("not a space on the board" in p for p in reading.problems)


def test_the_card_is_found_again_by_what_the_table_says():
    cards = [
        {"kind": "encounter", "number": 8, "space": "Rome", "deck": "Europe", "text": "the Rome part"},
        {"kind": "encounter", "number": 8, "space": "London", "deck": "Europe", "text": "the London part"},
        {"kind": "encounter", "number": 8, "space": "Arkham", "deck": "America", "text": "another deck"},
    ]
    assert dictate.find_encounter(8, space="Rome", cards=cards)["text"] == "the Rome part"
    assert dictate.find_encounter(8, space="Arkham", cards=cards)["deck"] == "America"
    assert dictate.find_encounter(9, space="Rome", cards=cards) is None


def test_the_agent_gets_the_card_or_an_honest_miss(monkeypatch):
    from src.speech import eleven_agent

    monkeypatch.setattr(
        dictate,
        "load",
        lambda: [
            {
                "kind": "encounter",
                "number": 8,
                "space": "Rome",
                "deck": "Europe",
                "name": "Europe 8, Rome",
                "text": "Test Influence.",
                "source_author": "Alessandro",
            }
        ],
    )
    found = eleven_agent.encounter_report(8, "Rome")
    assert found["known"] is True and found["text"] == "Test Influence."
    missing = eleven_agent.encounter_report(9, "Rome")
    assert missing["known"] is False
    assert missing["asked_for"] == "Europe 9, Rome"  # the deck is inferred from the city
    assert "read it out once" in missing["note"]
