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
