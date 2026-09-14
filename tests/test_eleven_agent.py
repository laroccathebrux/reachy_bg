import json

import pytest

from src.speech.eleven_agent import agent_config, knowledge_lookup, language_of, rules_lookup

VOICES = {"pt-BR": "voice-br", "en-US": "voice-us"}


def test_agent_config_has_native_voice_per_language_and_local_tools():
    cfg = agent_config(
        voices=VOICES, default_language="pt-BR", languages=("pt-BR", "en-US"), llm="", state_text=""
    )
    assert cfg["agent"]["language"] == "pt"
    assert cfg["tts"]["voice_id"] == "voice-br"
    assert cfg["tts"]["agent_output_audio_format"] == "pcm_16000"
    assert cfg["asr"]["user_input_audio_format"] == "pcm_16000"
    assert cfg["language_presets"]["en"]["overrides"]["tts"]["voice_id"] == "voice-us"
    assert cfg["language_presets"]["en"]["overrides"]["agent"]["language"] == "en"
    names = [t["name"] for t in cfg["agent"]["prompt"]["tools"]]
    assert names == ["game_rules", "game_knowledge", "language_detection"]
    assert "llm" not in cfg["agent"]["prompt"]
    assert "Golden rule" in cfg["agent"]["prompt"]["prompt"]


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
    with pytest.raises(ValueError):
        agent_config(voices={"pt-BR": "voice-br"}, default_language="pt-BR", languages=("pt-BR", "en-US"))


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
