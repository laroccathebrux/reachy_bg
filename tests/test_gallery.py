"""The gallery with a fake embedder (mean colour): add, cache, reload, match."""

import numpy as np

from src.vision.gallery import Gallery


class ColourEmbedder:
    """Unit vector of the mean BGR colour: enough to tell red pieces from blue ones."""

    def embed(self, images):
        out = []
        for img in images:
            v = img.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1.0
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def blob(colour, size=40):
    return np.full((size, size, 3), colour, dtype=np.uint8)


def test_add_match_and_cache(tmp_path):
    gallery = Gallery(tmp_path / "gallery", embedder=ColourEmbedder(), threshold=0.9)
    assert gallery.match(blob((0, 0, 200))) is None  # empty gallery
    gallery.add("investigator:Red", blob((0, 0, 200)), source="scan 1")
    gallery.add("gate:blue", blob((200, 30, 0)))
    gallery.add("gate:blue", blob((190, 40, 10)))
    assert gallery.counts() == {"investigator:Red": 1, "gate:blue": 2}
    match = gallery.match(blob((10, 5, 210)))
    assert match is not None and match.label == "investigator:Red" and match.kind == "investigator"
    assert match.name == "Red" and match.runner_up == "gate:blue" and match.samples == 1
    assert gallery.match(blob((100, 100, 100)), threshold=0.999) is None
    assert (tmp_path / "gallery" / "embeddings.npz").exists()

    # A fresh instance reads the cache without embedding again.
    class Counting(ColourEmbedder):
        calls = 0

        def embed(self, images):
            Counting.calls += 1
            return super().embed(images)

    again = Gallery(tmp_path / "gallery", embedder=Counting(), threshold=0.9)
    assert again.label_names == ["gate:blue", "investigator:Red"] and Counting.calls == 0
    assert again.match(blob((0, 0, 200))).label == "investigator:Red"
