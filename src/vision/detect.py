"""What is on the board that is not the board? Foreground pieces on the rectified map.

    baseline = Baseline.capture(registration, frame)          # the empty board, seen by this camera
    baseline.save(path); baseline = Baseline.load(path)
    pieces = find_pieces(registration, frame, baseline)       # -> [Piece(space="London", ...)]

The frame is warped onto the top-down map (:meth:`Registration.rectify`) and compared with a
baseline made the same way from the empty board at the start of the session. Same camera,
same light, same registration: the difference is what was put on the board (tokens, markers,
cards), and its centroid, mapped through the space table, names the space. This is the first,
class-free detector: it says *where* something is; naming *what* it is comes next (a gallery
of the real pieces). Each piece carries a crop of the original frame around it at full camera
resolution, the "zoom" the robot can show or match against the gallery.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger
from src.vision.board_map import Registration
from src.vision.spaces import nearest_space

log = get_logger(__name__)

MAP_WIDTH = 1200  # working width of the rectified map; the space table was annotated at this size
BLUR = 7  # pixels, on the rectified map, before differencing
# Measured on the empty board (two fresh frames against the baseline): the 99.9th percentile of
# the difference is 26-63 and the largest value 44-78; nothing false survives 60 with 300 px.
DIFF_THRESHOLD = 60  # summed absolute Lab difference (0..255 scale) that counts as "not the board"
MIN_AREA = 300  # rectified-map pixels: an investigator marker or a token is 400-1500 at 1200 wide
MAX_AREA = 40000
BOARD_MARGIN = 0.06  # fraction of the map height ignored at the top (Doom track) and bottom (Reserve)


@dataclass
class Piece:
    x: float  # rectified map pixels
    y: float
    width: float
    height: float
    area: int
    strength: float  # mean difference inside the blob
    space: str | None  # nearest space name, None when between spaces
    distance: float  # to the space centre, in board widths
    frame_box: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1 in the original frame
    crop: np.ndarray | None = field(default=None, repr=False)

    def record(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "area": self.area,
            "strength": round(self.strength, 1),
            "space": self.space,
            "distance": round(self.distance, 4),
            "frame_box": list(self.frame_box),
        }


@dataclass
class Baseline:
    """The empty board as this camera sees it, warped to the map, plus the mask of what the frame covered."""

    image: np.ndarray  # rectified, MAP_WIDTH wide, BGR
    coverage: np.ndarray  # uint8 mask: 255 where the camera frame covered the map
    captured_at: float = 0.0

    @classmethod
    def capture(
        cls,
        registration: Registration,
        frame: np.ndarray,
        width: int = MAP_WIDTH,
        *,
        extra: list[np.ndarray] = (),
    ) -> Baseline:
        """Rectify ``frame`` (and ``extra`` frames of the same view) and keep the per-pixel median."""
        rectified = [registration.rectify(f, width=width) for f in (frame, *extra)]
        image = (
            rectified[0] if len(rectified) == 1 else np.median(np.stack(rectified), axis=0).astype(np.uint8)
        )
        coverage = registration.rectify(np.full(frame.shape[:2], 255, dtype=np.uint8), width=width)
        return cls(image, coverage, time.time())

    def save(self, path: Path) -> Path:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.image)
        cv2.imwrite(str(path.with_name(path.stem + "_coverage.png")), self.coverage)
        path.with_suffix(".json").write_text(json.dumps({"captured_at": self.captured_at}), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Baseline:
        import cv2

        image = cv2.imread(str(path))
        coverage = cv2.imread(str(path.with_name(path.stem + "_coverage.png")), cv2.IMREAD_GRAYSCALE)
        if image is None or coverage is None:
            raise FileNotFoundError(f"baseline not found: {path}")
        meta = (
            json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if path.with_suffix(".json").exists()
            else {}
        )
        return cls(image, coverage, float(meta.get("captured_at", 0.0)))


def difference_map(rectified: np.ndarray, baseline: Baseline, *, blur: int = BLUR) -> np.ndarray:
    """Per-pixel difference (0..255*3) between the rectified frame and the baseline, in Lab after a blur."""
    import cv2

    k = max(1, blur | 1)
    a = cv2.cvtColor(cv2.GaussianBlur(rectified, (k, k), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
    b = cv2.cvtColor(cv2.GaussianBlur(baseline.image, (k, k), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
    diff = np.abs(a - b)
    # Lightness moves with the exposure; count it at half weight against colour changes.
    total = diff[:, :, 0] // 2 + diff[:, :, 1] + diff[:, :, 2]
    return np.clip(total, 0, 255 * 3).astype(np.uint16)


def find_pieces(
    registration: Registration,
    frame: np.ndarray,
    baseline: Baseline,
    *,
    threshold: int = DIFF_THRESHOLD,
    min_area: int = MIN_AREA,
    max_area: int = MAX_AREA,
    crop_margin: float = 0.6,
) -> list[Piece]:
    """Blobs of the rectified frame that differ from the baseline, each named by the nearest space."""
    import cv2

    width = baseline.image.shape[1]
    rectified = registration.rectify(frame, width=width)
    covered = registration.rectify(np.full(frame.shape[:2], 255, dtype=np.uint8), width=width)
    diff = difference_map(rectified, baseline)
    h, w = diff.shape
    mask = (diff >= threshold).astype(np.uint8) * 255
    mask[(covered < 255) | (baseline.coverage < 255)] = 0  # only where both pictures saw the board
    band = int(h * BOARD_MARGIN)
    mask[:band, :] = 0
    mask[h - band :, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pieces: list[Piece] = []
    scale_x = registration.reference_size[0] / width
    scale_y = registration.reference_size[1] / h
    fw, fh = registration.frame_size
    for i in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area or area > max_area:
            continue
        cx, cy = (float(v) for v in centroids[i])
        strength = float(diff[labels == i].mean())
        space, distance = nearest_space(cx / w, cy / h)
        # The blob's box back in the original frame, with a margin, for the zoomed crop.
        corners = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]) * [scale_x, scale_y]
        in_frame = registration.to_frame(corners)
        x0, y0 = in_frame.min(axis=0)
        x1, y1 = in_frame.max(axis=0)
        mx, my = (x1 - x0) * crop_margin, (y1 - y0) * crop_margin
        box = (
            int(max(0, x0 - mx)),
            int(max(0, y0 - my)),
            int(min(fw, x1 + mx)),
            int(min(fh, y1 + my)),
        )
        crop = frame[box[1] : box[3], box[0] : box[2]].copy() if box[2] > box[0] and box[3] > box[1] else None
        pieces.append(
            Piece(
                cx,
                cy,
                float(bw),
                float(bh),
                area,
                strength,
                space.name if space else None,
                distance,
                box,
                crop,
            )
        )
    pieces.sort(key=lambda p: -p.area)
    return pieces


def describe(pieces: list[Piece]) -> str:
    """One English sentence per piece, for the log and for the voice later."""
    if not pieces:
        return "Nothing on the board that is not the board."
    parts = []
    for piece in pieces:
        where = piece.space or "between spaces"
        parts.append(
            f"something at {where} ({int(piece.width)}x{int(piece.height)} px, strength {piece.strength:.0f})"
        )
    return "; ".join(parts) + "."


__all__ = ["Piece", "Baseline", "difference_map", "find_pieces", "describe", "MAP_WIDTH"]
