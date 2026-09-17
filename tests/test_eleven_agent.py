import json

import pytest

from src.speech.eleven_agent import agent_config, knowledge_lookup, language_of, rules_lookup

VOICES = {"pt-BR": "voice-br", "en-US": "voice-us"}


def _config(**kwargs):
    base = dict(voices=VOICES, default_language="pt-BR", languages=("pt-BR", "en-US"), llm="", state_text="")
    return agent_config(**{**base, **kwargs})


def test_agent_config_has_the_native_voice_and_the_local_tools():
    cfg = _config()
    assert cfg["agent"]["language"] == "pt"
    assert cfg["tts"]["voice_id"] == "voice-br"
    assert cfg["tts"]["agent_output_audio_format"] == "pcm_16000"
    assert cfg["asr"]["user_input_audio_format"] == "pcm_16000"
    names = [t["name"] for t in cfg["agent"]["prompt"]["tools"]]
    assert names == [
        "game_rules",
        "game_state",
        "take_turn",
        "remember_setup",
        "remember_note",
        "look_at_board",
        "card_value",
        "next_phase",
        "encounter_done",
        "encounter_card",
        "game_knowledge",
    ]
    assert "llm" not in cfg["agent"]["prompt"]
    assert "Golden rule" in cfg["agent"]["prompt"]["prompt"]


def test_the_locked_language_leaves_nothing_to_switch_with():
    cfg = _config(language_lock=True)
    names = [t["name"] for t in cfg["agent"]["prompt"]["tools"]]
    assert "language_detection" not in names  # no tool to switch with
    assert cfg["language_presets"] == {}  # and no other voice to switch to
    prompt = cfg["agent"]["prompt"]["prompt"]
    assert "You speak Brazilian Portuguese and only Brazilian Portuguese" in prompt
    assert "{language}" not in prompt


def test_unlocked_it_can_still_switch_the_way_it_used_to():
    cfg = _config(language_lock=False)
    names = [t["name"] for t in cfg["agent"]["prompt"]["tools"]]
    assert "language_detection" in names
    assert cfg["language_presets"]["en"]["overrides"]["tts"]["voice_id"] == "voice-us"
    assert cfg["language_presets"]["en"]["overrides"]["agent"]["language"] == "en"


def test_the_language_in_the_prompt_is_the_one_the_session_starts_in():
    assert "only American English" in _config(default_language="en-US")["agent"]["prompt"]["prompt"]


def test_agent_config_appends_state_and_llm():
    cfg = agent_config(
        voices=VOICES,
        default_language="pt-BR",
        languages=("pt-BR", "en-US"),
        llm="gpt-4.1-mini",
        state_text="Round 3.",
    )
    assert cfg["agent"]["prompt"]["prompt"].endswith("Round 3.")
    assert cfg["agent"]["prompt"]["llm"] == "gpt-4.1-mini"


def test_agent_config_refuses_a_language_without_a_native_voice():
    # Unlocked, the session may switch into English, so English needs its own voice.
    with pytest.raises(ValueError):
        agent_config(
            voices={"pt-BR": "voice-br"},
            default_language="pt-BR",
            languages=("pt-BR", "en-US"),
            language_lock=False,
        )
    # Locked, it never switches, so the missing voice is not this session's problem.
    cfg = agent_config(
        voices={"pt-BR": "voice-br"},
        default_language="pt-BR",
        languages=("pt-BR", "en-US"),
        language_lock=True,
    )
    assert cfg["tts"]["voice_id"] == "voice-br"


def fake_retriever(query, *, rules_limit, knowledge_limit):
    if "nothing" in query:
        return []
    return [
        {
            "source": "rulebook",
            "path": "Reserve",
            "page_start": 6,
            "text": f"Passage for {query}",
            "score": 0.7,
        }
    ]


def test_rules_lookup_formats_passages_and_note():
    result = rules_lookup("How many actions?", retriever=fake_retriever)
    assert "[1] (rulebook: Reserve, page 6)" in result["passages"]
    assert "ONLY these passages" in result["note"]
    empty = rules_lookup("nothing", retriever=fake_retriever)
    assert empty["passages"] == "(no passages found)"
    assert "could not find" in empty["note"]


def test_knowledge_lookup_uses_the_name():
    result = knowledge_lookup("Lily Chen", "starting possessions", retriever=fake_retriever)
    assert "Lily Chen starting possessions" in result["passages"]


def test_language_of():
    assert language_of("pt") == "pt-BR"
    assert language_of("en") == "en-US"
    assert language_of("xx") == "pt-BR"


def test_client_tools_stop_searching_once_the_session_is_closing():
    from src.speech.eleven_agent import CLOSING_RESULT, client_tools

    calls = []
    alive = {"ok": True}

    def retriever(question, **kwargs):
        calls.append(question)
        return []

    tools = client_tools(retriever=retriever, active=lambda: alive["ok"])
    handler, _ = tools.tools["game_rules"]
    handler({"question": "how many actions?"})
    assert calls == ["how many actions?"]
    alive["ok"] = False
    assert json.loads(handler({"question": "again?"})) == CLOSING_RESULT
    assert calls == ["how many actions?"]


def test_the_agent_can_read_the_game_the_robot_remembers():
    from src.speech.eleven_agent import game_state_report
    from src.strategy.game import ROBOT, GameState

    assert game_state_report(None)["known"] is False
    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Jacqueline Fine", controller="Alessandro")
    game.reserve = ["Bull Whip"]
    report = game_state_report(game)
    assert report["ancient_one"] == "Azathoth" and report["doom"] == 15
    assert report["my_investigator"]["name"] == "Lily Chen"
    assert {i["player"] for i in report["investigators"]} == {"me", "Alessandro"}
    assert report["reserve"] == ["Bull Whip"]
    assert any("Mystery" in m for m in report["still_missing"])


def test_the_game_state_tool_reads_the_game_at_call_time():
    import json

    from src.speech.eleven_agent import client_tools
    from src.strategy.game import ROBOT, GameState

    game = GameState()
    tools = client_tools(game=lambda: game)
    handler, _ = tools.tools["game_state"]
    before = json.loads(handler({}))
    game.set_ancient_one("Azathoth")  # the setup happens while the session is already running
    game.add_investigator("Lily Chen", controller=ROBOT)
    after = json.loads(handler({}))
    assert before["ancient_one"] == ""
    assert after["ancient_one"] == "Azathoth"


def test_the_quiet_rule_goes_away_with_the_local_gate():
    """With the gate off every sentence at the table reaches the agent, so leaving its own
    "otherwise stay quiet" rule in the prompt gates the robot twice - which is how a question
    asked straight at it ("Tu nao consegue ver o tabuleiro?") was ignored on 2026-09-17."""
    gated = _config()["agent"]["prompt"]["prompt"]
    assert "Otherwise stay quiet." in gated
    assert "{addressing}" not in gated

    open_ = _config(answer_everything=True)["agent"]["prompt"]["prompt"]
    assert "Otherwise stay quiet." not in open_
    assert "never stay silent because your name was not said" in open_
    assert "{addressing}" not in open_


def test_the_prompt_forbids_claiming_a_note_that_was_never_written():
    prompt = _config()["agent"]["prompt"]["prompt"]
    assert "remember_note" in prompt
    assert "Never say you have written something down" in prompt


def test_the_prompt_no_longer_says_it_cannot_see_the_board():
    """Two days of board detection, and the prompt still opened with "You cannot see the
    board" - so on 2026-09-17 she answered "Não consigo ver o tabuleiro" to a direct question,
    which was true of the process she was speaking from and false of the robot."""
    prompt = _config()["agent"]["prompt"]["prompt"]
    assert "You cannot see the board" not in prompt
    assert "never say you cannot see the board" in prompt
    assert "look_at_board" in prompt
    names = [t["name"] for t in _config()["agent"]["prompt"]["tools"]]
    assert "look_at_board" in names


def test_game_state_carries_one_line_of_board_and_the_tool_has_the_detail():
    from src.speech.eleven_agent import game_state_report
    from src.strategy.game import GameState

    report = game_state_report(GameState(), "the camera sees 2 piece(s); known: Lily Chen at Rome")
    assert report["board"] == "the camera sees 2 piece(s); known: Lily Chen at Rome"
    assert "look_at_board" in report["note"]
    assert game_state_report(GameState())["board"] == ""  # no camera, no claim


def test_a_value_read_out_at_the_table_stops_acquire_assets_asking_again(tmp_path):
    """2026-09-17: "Lucky Cigarette Case é dois" was heard, thanked for and forgotten, because
    card values could only arrive through the offline dictate script. Forty seconds later the
    robot asked for the same numbers: "Eu acabei de falar." """
    from src.rag.dictate import remember_value
    from src.strategy import moves

    cards = tmp_path / "cards.json"
    remember_value("Lucky Cigarette Case", 2, path=cards)
    remember_value("Kerosene", 1, path=cards)

    saved = {c["name"]: c["value"] for c in __import__("json").loads(cards.read_text())}
    assert saved == {"Lucky Cigarette Case": 2, "Kerosene": 1}
    # card_value reads through an lru_cache, so a mid-session write has to drop it or the
    # process that wrote the number still cannot see it.
    assert moves._dictated.cache_info().currsize == 0


def test_game_state_says_where_the_round_is():
    from src.speech.eleven_agent import game_state_report
    from src.strategy.game import ROBOT, GameState

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    assert "has not started" in game_state_report(game)["where_we_are"]
    game.begin_round()
    assert "round 1, Action Phase" in game_state_report(game)["where_we_are"]


def test_a_card_is_found_by_its_name_before_the_vector_search():
    """2026-09-17: "Não encontrei a descrição do efeito do Witch Doctor" - while the card sat in
    the repository with its text. game_knowledge went to the embeddings, which are worst at
    exactly this: "Witch Doctor" came back as Diana Stanley and the Witch monster."""
    from src.speech.eleven_agent import _card_by_name, knowledge_lookup

    assert "recover" in _card_by_name("Witch Doctor")
    # The box hyphenates it and the table does not; punctuation and case must not decide.
    assert _card_by_name("Double Barreled Shotgun") == _card_by_name("double-barreled shotgun")
    assert _card_by_name("Double Barreled Shotgun")
    assert _card_by_name("a card this box does not have") == ""

    found = knowledge_lookup("Witch Doctor", retriever=lambda *a, **k: [])
    assert "printed on the card" in found["passages"]
    assert "recover" in found["passages"]

    missing = knowledge_lookup("Nonesuch", retriever=lambda *a, **k: [])
    assert "printed on the card" not in missing["passages"]
