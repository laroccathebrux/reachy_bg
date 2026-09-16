"""The 16 Mysteries of the box: checked data, and the destination they put on the map."""

from __future__ import annotations

from src.rag import ingest_mysteries as mysteries_ingest
from src.strategy import decide, moves
from src.strategy.game import ROBOT, GameState
from src.strategy.reference import ANCIENT_ONES, closest_mysteries, mysteries, mystery
from src.vision.spaces import BY_NAME


def test_the_list_is_four_mysteries_for_each_of_the_four_ancient_ones():
    entries = mysteries_ingest.load()
    assert mysteries_ingest.check(entries) == []
    assert len(entries) == len(ANCIENT_ONES) * mysteries_ingest.PER_ANCIENT_ONE


def test_every_space_a_mystery_names_is_a_space_on_the_board():
    for found in mysteries():
        for space in found.spaces:
            assert space in BY_NAME, f"{found.name} names {space}"


def test_a_mystery_is_found_however_it_is_written():
    assert mystery("the deep ones attack!").ancient_one == "Cthulhu"
    assert mystery("THE TRUE NAME").ancient_one == "Azathoth"
    assert mystery("Nyarlathotep's Revenge") is None  # not in this box


def test_a_mangled_name_gets_its_candidates_back():
    # What Whisper actually wrote in the live sessions.
    assert "The Deep Ones Attack!" in closest_mysteries("The Deepest One Attack")
    assert "Rituals in the Wild" in closest_mysteries("Rituais na natureza")
    assert closest_mysteries("Q-Mystery") == []  # too far to guess, so it asks


def test_each_record_carries_its_source_and_reads_as_a_fact():
    for record in mysteries_ingest.records():
        assert record["kind"] == "mystery" and record["base_game"] is True
        assert record["source_url"].startswith("https://")
        assert record["ancient_one"] in {one.name for one in ANCIENT_ONES}
        assert record["text"].startswith(f"Mystery of {record['ancient_one']}")


def test_an_epic_monster_mystery_names_the_monster_and_where_it_spawns():
    found = mystery("Spawn of Yog-Sothoth")
    assert found.epic_monster == "Dunwich Horror"
    assert found.spaces == ("Arkham",)


def _game(space: str = "Shanghai") -> GameState:
    game = GameState()
    game.set_ancient_one("Shub-Niggurath")
    game.add_investigator("Lily Chen", controller=ROBOT, space=space)
    return game


def test_the_active_mystery_becomes_a_destination_on_the_map():
    game = _game()
    assert decide.goals(game) == {}
    game.mystery = "Rituals in the Wild"
    found = decide.goals(game)
    assert set(found) == {"4", "10", "21", "Tunguska"}
    assert "the Mystery Rituals in the Wild" in found["Tunguska"]


def test_walking_towards_the_mystery_scores_better_than_walking_away(monkeypatch):
    monkeypatch.setattr(decide.map_graph, "VERIFIED", True)
    game = _game()
    game.mystery = "Rituals in the Wild"  # 4, 10, 21, Tunguska
    situation = moves.situation_of(game, game.robot_investigator)
    plans = {p.ends_at: p for p in moves.plans(game, situation) if len(p.actions) == 1}
    towards = decide.score(game, situation, plans["19"])  # 19 is next to Tunguska
    away = decide.score(game, situation, plans["20"])
    assert towards.score > away.score
    assert any("Rituals in the Wild" in r for r in towards.reasons)


def test_the_brief_tells_the_model_what_solving_it_takes():
    game = _game()
    game.mystery = "Spawn of the Black Goat"
    text = decide.brief(game, moves.situation_of(game, game.robot_investigator))
    assert "Defeat the Nug Epic Monster" in text
    assert "It is at The Amazon" in text


def test_a_mystery_the_box_does_not_have_is_left_as_it_was_said():
    game = _game()
    game.mystery = "Q-Mystery"  # what Whisper left behind in the live session
    text = decide.brief(game, moves.situation_of(game, game.robot_investigator))
    assert "Mystery: Q-Mystery" in text
    assert decide.goals(game) == {}
