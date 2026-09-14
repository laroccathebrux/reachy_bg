import importlib

import src.config as config


def test_defaults_are_sane():
    assert config.GAME_ID == "eldritch-horror"
    assert config.QDRANT_URL.startswith("http")
    assert config.OLLAMA_BASE_URL.startswith("http")
    assert config.EMBED_MODEL == "bge-m3"
    assert config.REACHY_PORT == 8000
    assert config.AUDIO_SAMPLE_RATE == 16000


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("REACHY_PORT", "8001")
    monkeypatch.setenv("DEBUG", "true")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.OLLAMA_MODEL == "qwen2.5:3b"
        assert reloaded.REACHY_PORT == 8001
        assert reloaded.DEBUG is True
    finally:
        monkeypatch.delenv("OLLAMA_MODEL")
        monkeypatch.delenv("REACHY_PORT")
        monkeypatch.delenv("DEBUG")
        importlib.reload(config)


def test_validate_config_reports_missing_tts_key(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    reloaded = importlib.reload(config)
    try:
        assert any("ELEVENLABS_API_KEY" in p for p in reloaded.validate_config())
    finally:
        monkeypatch.delenv("TTS_PROVIDER")
        monkeypatch.delenv("ELEVENLABS_API_KEY")
        importlib.reload(config)


def test_import_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    reloaded = importlib.reload(config)
    try:
        assert not (tmp_path / "data").exists()
        reloaded.ensure_data_dirs()
        assert (tmp_path / "data" / "captures").is_dir()
    finally:
        monkeypatch.delenv("DATA_DIR")
        importlib.reload(config)


def test_bilingual_defaults_and_voice_fallback(monkeypatch):
    monkeypatch.setenv("SPOKEN_LANGUAGES", "pt-BR,en-US")
    monkeypatch.setenv("DEFAULT_LANGUAGE", "pt-BR")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID_PT_BR", "voice-br")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID_EN_US", "voice-us")
    monkeypatch.setenv("LOCAL_TTS_VOICE_PT_BR", "pt_BR-faber-medium")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.SPOKEN_LANGUAGES == ("pt-BR", "en-US")
        assert reloaded.voice_for("pt-BR") == "voice-br"
        assert reloaded.voice_for("en-US") == "voice-us"
        # A language without a local voice falls back to the default language's voice.
        assert reloaded.voice_for("en-US", provider="local") == "pt_BR-faber-medium"
        assert reloaded.voice_for("fr-FR") == "voice-br"
    finally:
        for name in (
            "SPOKEN_LANGUAGES",
            "DEFAULT_LANGUAGE",
            "ELEVENLABS_VOICE_ID_PT_BR",
            "ELEVENLABS_VOICE_ID_EN_US",
            "LOCAL_TTS_VOICE_PT_BR",
        ):
            monkeypatch.delenv(name)
        importlib.reload(config)


def test_validate_config_requires_a_native_voice_per_language(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID_PT_BR", "voice-br")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID_EN_US", raising=False)
    reloaded = importlib.reload(config)
    try:
        # Falling back to the Brazilian voice for English is exactly what must not happen
        # silently: voice_for() degrades gracefully, but the validator reports it.
        problems = reloaded.validate_config()
        assert any("en-US" in p for p in problems)
        assert not any("pt-BR" in p for p in problems)
        assert reloaded.voice_for("en-US") == "voice-br"
    finally:
        for name in ("TTS_PROVIDER", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID_PT_BR"):
            monkeypatch.delenv(name)
        importlib.reload(config)
