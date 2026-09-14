"""Who is talking: voiceprints per player, matched by cosine similarity.

    registry = SpeakerRegistry.load()                 # data/speakers/voiceprints.json
    registry.enroll("Alessandro", utterance.audio)    # one sentence per player at game start
    name, score = registry.identify(utterance.audio)  # ("Alessandro", 0.52) or ("", 0.12)

Embeddings come from ``pyannote/wespeaker-voxceleb-resnet34-LM`` (256-d, language-agnostic,
ungated) through ``pyannote.audio`` in this process; ``Embedder`` loads it lazily so the
registry logic is testable with fake vectors.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import HF_TOKEN, SPEAKER_DIR, SPEAKER_EMBEDDING_MODEL, SPEAKER_MATCH_THRESHOLD
from src.logger import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16_000
MIN_ENROLL_S = 1.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


class Embedder:
    """Speaker embedding of mono int16/float audio at 16 kHz."""

    def __init__(self, model: str = SPEAKER_EMBEDDING_MODEL):
        self.model_name = model
        self._inference: Any | None = None

    def _load(self) -> Any:
        if self._inference is None:
            import torch  # noqa: F401  (pyannote needs it imported first on macOS)
            from huggingface_hub import get_token
            from pyannote.audio import Inference, Model

            started = time.perf_counter()
            model = Model.from_pretrained(self.model_name, token=HF_TOKEN or get_token())
            self._inference = Inference(model, window="whole")
            log.info("speaker model %s loaded in %.1fs", self.model_name, time.perf_counter() - started)
        return self._inference

    def __call__(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        import torch

        if audio.dtype.kind in "iu":
            audio = audio.astype(np.float32) / 32768.0
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        embedding = self._load()({"waveform": waveform, "sample_rate": sample_rate})
        return np.asarray(embedding, dtype=np.float32).ravel()


@dataclass
class SpeakerRegistry:
    """Named voiceprints (several embeddings per name) with persistence."""

    threshold: float = SPEAKER_MATCH_THRESHOLD
    embedder: Any = None  # callable audio -> vector; defaults to Embedder()
    prints: dict[str, list[np.ndarray]] = field(default_factory=dict)
    path: Path = SPEAKER_DIR / "voiceprints.json"

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path | None = None, **kwargs: Any) -> SpeakerRegistry:
        registry = cls(**kwargs)
        if path is not None:
            registry.path = path
        if registry.path.exists():
            data = json.loads(registry.path.read_text(encoding="utf-8"))
            registry.prints = {
                name: [np.asarray(v, dtype=np.float32) for v in vecs] for name, vecs in data.items()
            }
            log.info("voiceprints loaded for %s", ", ".join(registry.prints) or "nobody")
        return registry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: [np.round(v, 5).tolist() for v in vecs] for name, vecs in self.prints.items()}
        self.path.write_text(json.dumps(data), encoding="utf-8")

    # ------------------------------------------------------------------ core
    @property
    def names(self) -> list[str]:
        return list(self.prints)

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        if self.embedder is None:
            self.embedder = Embedder()
        return self.embedder(audio)

    def enroll(self, name: str, audio: np.ndarray, *, replace: bool = False) -> None:
        if audio.size < MIN_ENROLL_S * SAMPLE_RATE:
            raise ValueError(f"enrolment clip for {name} is shorter than {MIN_ENROLL_S}s")
        vector = self._embed(audio)
        if replace or name not in self.prints:
            self.prints[name] = [vector]
        else:
            self.prints[name].append(vector)
        log.info("voiceprint enrolled for %s (%d clips)", name, len(self.prints[name]))

    def scores(self, vector: np.ndarray) -> dict[str, float]:
        """Best cosine similarity per enrolled name."""
        return {name: max(cosine(vector, v) for v in vecs) for name, vecs in self.prints.items() if vecs}

    def identify(self, audio: np.ndarray) -> tuple[str, float]:
        """``(name, score)`` for the closest voiceprint, or ``("", best score)`` below the threshold."""
        if not self.prints:
            return "", 0.0
        scores = self.scores(self._embed(audio))
        name, score = max(scores.items(), key=lambda kv: kv[1])
        return (name if score >= self.threshold else ""), round(score, 3)


__all__ = ["Embedder", "SpeakerRegistry", "cosine", "MIN_ENROLL_S"]
