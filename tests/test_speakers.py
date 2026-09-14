import numpy as np
import pytest

from src.speech.speakers import SpeakerRegistry, cosine

RATE = 16_000


def fake_embedder(audio: np.ndarray) -> np.ndarray:
    """A 'voice' is the mean sample value; the embedding is a 4-d one-hot-ish vector around it."""
    level = float(np.mean(audio))
    return np.array([1.0, level / 100.0, (level / 100.0) ** 2, 0.1], dtype=np.float32)


def clip(level: int, seconds: float = 1.5) -> np.ndarray:
    return np.full(int(RATE * seconds), level, dtype=np.int16)


def test_cosine():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert cosine(np.zeros(2), np.ones(2)) == 0.0


def test_enroll_identify_and_unknown(tmp_path):
    registry = SpeakerRegistry(threshold=0.99, embedder=fake_embedder, path=tmp_path / "v.json")
    assert registry.identify(clip(100)) == ("", 0.0)
    registry.enroll("Ana", clip(100))
    registry.enroll("Bruno", clip(-100))
    assert registry.identify(clip(100)) == ("Ana", 1.0)
    assert registry.identify(clip(-100)) == ("Bruno", 1.0)
    name, score = registry.identify(clip(0))
    assert name == "" and score < 0.99


def test_multiple_clips_per_name_and_persistence(tmp_path):
    path = tmp_path / "v.json"
    registry = SpeakerRegistry(threshold=0.5, embedder=fake_embedder, path=path)
    registry.enroll("Ana", clip(100))
    registry.enroll("Ana", clip(120))
    assert len(registry.prints["Ana"]) == 2
    registry.enroll("Ana", clip(90), replace=True)
    assert len(registry.prints["Ana"]) == 1
    registry.save()
    loaded = SpeakerRegistry.load(path, threshold=0.5, embedder=fake_embedder)
    assert loaded.names == ["Ana"]
    assert loaded.identify(clip(90))[0] == "Ana"


def test_enrolment_needs_a_full_second():
    registry = SpeakerRegistry(embedder=fake_embedder)
    with pytest.raises(ValueError):
        registry.enroll("Ana", clip(100, seconds=0.5))
