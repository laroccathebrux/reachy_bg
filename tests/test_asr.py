"""Language mapping and hallucination filtering (pure), plus a real Whisper run when available."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.speech.asr import (
    choose_language,
    is_hallucination,
    map_language,
    normalise,
    summarise_segments,
    to_float32,
    whisper_code,
)

SUPPORTED = ("pt-BR", "en-US")


def test_map_language_to_configured_tags():
    assert map_language("pt", SUPPORTED) == "pt-BR"
    assert map_language("en", SUPPORTED) == "en-US"
    assert map_language("PT-pt", SUPPORTED) == "pt-BR"
    assert map_language("es", SUPPORTED) == ""
    assert map_language("", SUPPORTED) == ""


def test_whisper_code():
    assert whisper_code("pt-BR") == "pt"
    assert whisper_code("en") == "en"
    assert whisper_code("auto") == ""
    assert whisper_code("") == ""


def test_choose_language_restricts_to_supported():
    probs = {"gl": 0.45, "pt": 0.40, "es": 0.10, "en": 0.05}
    assert choose_language(probs, SUPPORTED) == ("pt-BR", 0.40)
    assert choose_language({"de": 0.9, "fr": 0.1}, SUPPORTED) == ("", 0.0)
    assert choose_language({"en": 0.7, "pt": 0.3}, SUPPORTED) == ("en-US", 0.7)


def test_hallucination_filter():
    assert is_hallucination("Obrigado.", -0.3, 0.1)
    assert is_hallucination("Legendas pela comunidade Amara.org", -0.3, 0.1)
    assert is_hallucination("", -0.3, 0.1)
    assert is_hallucination("whatever it said", -1.5, 0.9)
    assert not is_hallucination("whatever it said", -0.4, 0.9)
    assert not is_hallucination("Quantas ações por rodada?", -0.2, 0.05)
    assert is_hallucination("Cough, Cough, Cough.", -0.3, 0.2)
    assert is_hallucination("Hmm.", -0.3, 0.2)
    assert is_hallucination("Eeeeeeeeeeeeeeeeeeeeeeeeeeee", -0.3, 0.2)
    assert not is_hallucination("Reeeally? Can I rest?", -0.3, 0.2)
    assert not is_hallucination("Hmm, can I rest?", -0.3, 0.2)


def test_normalise():
    assert normalise("  Thank you!! ") == "thank you"


def test_summarise_segments_weights_by_duration():
    segments = [
        {"start": 0, "end": 1, "avg_logprob": -0.2, "no_speech_prob": 0.1},
        {"start": 1, "end": 4, "avg_logprob": -0.6, "no_speech_prob": 0.3},
    ]
    avg, no_speech = summarise_segments(segments)
    assert abs(avg - (-0.5)) < 1e-9
    assert no_speech == 0.3
    assert summarise_segments([]) == (-1.0, 1.0)


def test_to_float32():
    out = to_float32(np.array([0, 16384, -32768], dtype=np.int16))
    assert out.dtype == np.float32
    assert np.allclose(out, [0.0, 0.5, -1.0])


# --------------------------------------------------------------------------- integration
def _model_cached() -> bool:
    try:
        import mlx_whisper  # noqa: F401
        from huggingface_hub import try_to_load_from_cache

        from src.config import WHISPER_MODEL

        return isinstance(try_to_load_from_cache(WHISPER_MODEL, "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(
    not _model_cached() or shutil.which("say") is None, reason="Whisper model or macOS `say` unavailable"
)
def test_whisper_transcribes_a_synthetic_english_sentence(tmp_path: Path):
    from src.speech.asr import Transcriber
    from src.speech.microphone import read_wav

    wav = tmp_path / "q.wav"
    subprocess.run(
        ["say", "-o", str(wav), "--data-format=LEI16@16000", "How many actions can I take per round?"],
        check=True,
    )
    transcriber = Transcriber(supported=SUPPORTED)
    result = transcriber.transcribe(read_wav(wav))
    assert result.language == "en-US"
    assert result.language_confidence > 0.5
    assert "actions" in result.text.lower()
    assert result.seconds < 30


def test_looping_keywords_are_a_hallucination():
    from src.speech.asr import is_hallucination

    assert is_hallucination("Omen, omen, omen, omen, omen, omen, omen.", -0.2, 0.1)
    assert not is_hallucination("How many actions can I take per round?", -0.2, 0.1)
    assert not is_hallucination("omen omen", -0.2, 0.1)  # too short to judge by repetition
