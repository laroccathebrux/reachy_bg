"""Naming pieces: a gallery of the real pieces, photographed by this camera and labelled by the owner.

    gallery = Gallery()                       # GALLERY_DIR/<label>/*.jpg, embeddings cached next to them
    gallery.add("investigator:Akachi Onyele", crop)
    match = gallery.match(crop)               # Match(label, score, runner_up) or None below the threshold

Every crop is embedded with a pretrained ResNet-50 (torchvision, ImageNet weights, the 2048-d
pooled features) and compared by cosine similarity; the label of the nearest gallery crop
wins when its score passes ``GALLERY_MATCH_THRESHOLD``. Labels are free text with a kind
prefix: ``investigator:Name``, ``asset:Name``, ``gate``, ``clue``, ``monster:Name``, ...
The embedder is injectable, so the gallery logic is tested without torch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import GALLERY_DIR, GALLERY_MATCH_THRESHOLD
from src.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Match:
    label: str
    score: float
    runner_up: str | None = None
    runner_up_score: float = 0.0
    samples: int = 0

    @property
    def kind(self) -> str:
        return self.label.split(":", 1)[0]

    @property
    def name(self) -> str:
        return self.label.split(":", 1)[1] if ":" in self.label else self.label


class ResNetEmbedder:
    """2048-d pooled features of a pretrained ResNet-50; BGR uint8 crops in, unit vectors out."""

    def __init__(self) -> None:
        self._model: Any = None
        self._device: Any = None

    def _load(self) -> None:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        model.eval()
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = model.to(self._device)
        log.info("gallery embedder: ResNet-50 on %s", self._device)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        import cv2
        import torch

        if self._model is None:
            self._load()
        batch = []
        for image in images:
            rgb = cv2.cvtColor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
                [0.229, 0.224, 0.225]
            )[:, None, None]
            batch.append(x)
        with torch.no_grad():
            features = self._model(torch.stack(batch).to(self._device)).float().cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
        return (features / norms).astype(np.float32)


class Gallery:
    def __init__(
        self,
        root: Path | None = None,
        *,
        embedder: Any = None,
        threshold: float = GALLERY_MATCH_THRESHOLD,
    ):
        self.root = Path(root or GALLERY_DIR)
        self.embedder = embedder or ResNetEmbedder()
        self.threshold = threshold
        self.paths: list[Path] = []
        self.labels: list[str] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.reload()

    # ------------------------------------------------------------------ contents
    @property
    def cache_path(self) -> Path:
        return self.root / "embeddings.npz"

    def reload(self) -> None:
        """Index every ``<label>/<file>.jpg``; embeddings are cached by path and modification time."""
        import cv2

        files = sorted(p for p in self.root.glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        cached: dict[str, tuple[float, np.ndarray]] = {}
        if self.cache_path.exists():
            try:
                data = np.load(self.cache_path, allow_pickle=False)
                for key, mtime, vec in zip(data["keys"], data["mtimes"], data["vectors"], strict=True):
                    cached[str(key)] = (float(mtime), vec)
            except (OSError, ValueError, KeyError) as exc:
                log.warning("gallery cache unreadable (%s); recomputing", exc)
        vectors: list[np.ndarray] = []
        todo: list[tuple[int, Path]] = []
        for i, path in enumerate(files):
            key = str(path.relative_to(self.root))
            mtime = path.stat().st_mtime
            hit = cached.get(key)
            if hit is not None and abs(hit[0] - mtime) < 1e-6:
                vectors.append(hit[1])
            else:
                vectors.append(None)  # type: ignore[arg-type]
                todo.append((i, path))
        if todo:
            images = [cv2.imread(str(path)) for _, path in todo]
            good = [(i, img) for (i, _), img in zip(todo, images, strict=True) if img is not None]
            if good:
                embedded = self.embedder.embed([img for _, img in good])
                for (i, _), vec in zip(good, embedded, strict=True):
                    vectors[i] = vec
        keep = [i for i, v in enumerate(vectors) if v is not None]
        self.paths = [files[i] for i in keep]
        self.labels = [files[i].parent.name for i in keep]
        self.vectors = (
            np.stack([vectors[i] for i in keep]).astype(np.float32) if keep else np.zeros((0, 0), np.float32)
        )
        if todo and keep:
            self._save_cache()
        log.info("gallery: %d crops, %d labels", len(self.paths), len(set(self.labels)))

    def _save_cache(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.cache_path,
            keys=np.array([str(p.relative_to(self.root)) for p in self.paths]),
            mtimes=np.array([p.stat().st_mtime for p in self.paths], dtype=np.float64),
            vectors=self.vectors,
        )

    @property
    def label_names(self) -> list[str]:
        return sorted(set(self.labels))

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for label in self.labels:
            result[label] = result.get(label, 0) + 1
        return result

    # ------------------------------------------------------------------ use
    def add(self, label: str, image: np.ndarray, *, source: str = "") -> Path:
        """Save a crop under ``label`` and index it."""
        import cv2

        label = label.strip().replace("/", "-")
        if not label:
            raise ValueError("empty label")
        folder = self.root / label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{len(list(folder.glob('*.jpg'))) + 1:03d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        vec = self.embedder.embed([image])[0].astype(np.float32)
        self.paths.append(path)
        self.labels.append(label)
        self.vectors = np.vstack([self.vectors, vec[None]]) if self.vectors.size else vec[None]
        self._save_cache()
        (folder / "sources.jsonl").open("a", encoding="utf-8").write(
            json.dumps({"file": path.name, "source": source}) + "\n"
        )
        log.info("gallery: %s <- %s", label, path.name)
        return path

    def match(self, image: np.ndarray, *, threshold: float | None = None) -> Match | None:
        """Nearest label by cosine similarity (best crop per label), or None below the threshold."""
        if not self.labels:
            return None
        vec = self.embedder.embed([image])[0]
        scores = self.vectors @ vec
        best_per_label: dict[str, float] = {}
        for label, score in zip(self.labels, scores, strict=True):
            best_per_label[label] = max(best_per_label.get(label, -1.0), float(score))
        ranked = sorted(best_per_label.items(), key=lambda kv: -kv[1])
        label, score = ranked[0]
        limit = self.threshold if threshold is None else threshold
        if score < limit:
            return None
        runner = ranked[1] if len(ranked) > 1 else (None, 0.0)
        return Match(label, round(score, 3), runner[0], round(runner[1], 3), self.labels.count(label))


__all__ = ["Gallery", "Match", "ResNetEmbedder"]
