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
CLOSER_LOOK_STRENGTH = 60.0  # kept for callers that pass an absolute limit
CLOSER_LOOK_MARGIN = 15.0  # a sighting weaker than threshold + margin gets a closer look (owner's rule)
EDGE_FRACTION = 0.12  # of the frame width/height: the lens distorts there, the homography does not
BLUR = 7  # pixels, on the rectified map, before differencing
# Measured on the empty board against a median baseline: no false blob of 300 px survives 45,
# while a dark investigator standee on the dark green Buenos Aires circle scores 53 (1200 px).
DIFF_THRESHOLD = 45  # upper bound of the adaptive threshold (evening light, noisy baseline)
DIFF_THRESHOLD_MIN = 25  # lower bound (flat daylight: standees differ by only 33-40, noise p90 is 7)
NOISE_FACTOR = 3.0  # threshold = NOISE_FACTOR x the 95th percentile of the difference over the board
MIN_AREA = 300  # rectified-map pixels: an investigator marker or a token is 400-1500 at 1200 wide
NEAR_FACTOR = 2.8  # a piece this many radii from a space centre is reported as "near" it
MAX_AREA = 40000
STANDEE_ASPECT = 1.25  # frame height over width from which a blob is treated as a standing piece
STANDEE_DROP = 0.35  # of the blob's frame height: how far below the visible bottom the base is put
MIN_SIDE = (
    20  # rectified-map pixels: thinner blobs are slivers along the frame or board edges (a token is 25+)
)
BOARD_MARGIN_TOP = 0.09  # fraction of the map height ignored at the top: the Doom track and hands beyond it
BOARD_MARGIN_BOTTOM = 0.06  # and at the bottom edge
RESERVE_BOX = (0.0, 0.82, 0.37, 1.0)  # x0, y0, x1, y1 of the Reserve (cards on it are read separately)
LEGEND_BOX = (0.0, 0.52, 0.17, 0.71)  # the legend box: nothing is ever placed on it


@dataclass
class Piece:
    x: float  # rectified map pixels
    y: float
    width: float
    height: float
    area: int
    strength: float  # mean difference inside the blob
    space: str | None  # the space the piece stands on, None when it is only near one (or none)
    distance: float  # to the nearest space centre, in board widths
    near: str | None = None  # nearest space when not on one
    frame_box: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1 in the original frame
    crop: np.ndarray | None = field(default=None, repr=False)
    views: list[str] = field(default_factory=list)  # which views of a scan saw it
    frame_base: tuple[float, float] = (0.0, 0.0)  # frame pixel where the piece touches the board
    confirmed: bool = False  # verdict taken from a closer look with the piece at the frame centre
    radius_ratio: float = 0.0  # distance to the space centre over the space radius (0 = dead centre)
    edge: bool = False  # base within EDGE_FRACTION of the frame border (lens distortion zone)
    kind: str = "piece"  # piece | die
    value: int | None = None  # what a die shows
    kind_confidence: float = 0.0
    sighting_crops: list[np.ndarray] = field(default_factory=list, repr=False)  # every look at it
    threshold: float = DIFF_THRESHOLD  # the difference threshold this sighting was found with

    def needs_closer_look(self, min_strength: float | None = None) -> bool:
        """Ambiguous enough to be worth pointing the camera straight at it."""
        limit = self.threshold + CLOSER_LOOK_MARGIN if min_strength is None else min_strength
        weak = self.strength < limit
        return not self.confirmed and (self.space is None or self.radius_ratio > 0.6 or self.edge or weak)

    def record(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "area": self.area,
            "strength": round(self.strength, 1),
            "space": self.space,
            "near": self.near,
            "distance": round(self.distance, 4),
            "frame_box": list(self.frame_box),
            "views": list(self.views),
            "frame_base": [round(self.frame_base[0], 1), round(self.frame_base[1], 1)],
            "confirmed": self.confirmed,
            "edge": self.edge,
            "kind": self.kind,
            "value": self.value,
            "kind_confidence": round(self.kind_confidence, 2),
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


class BaselineSet:
    """One baseline per view of a scan (centre, left, right), saved as ``<folder>/<view>.jpg``."""

    def __init__(self) -> None:
        self.views: dict[str, Baseline] = {}

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for name, baseline in self.views.items():
            baseline.save(folder / f"{name}.jpg")
        (folder / "views.json").write_text(json.dumps(sorted(self.views)), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> BaselineSet:
        index = folder / "views.json"
        if not index.exists():
            raise FileNotFoundError(f"no baseline set in {folder}")
        result = cls()
        for name in json.loads(index.read_text(encoding="utf-8")):
            result.views[name] = Baseline.load(folder / f"{name}.jpg")
        return result


MERGE_RADIUS = 30.0  # rectified-map pixels: sightings closer than this from two views are one piece


def merge_pieces(sightings: list[tuple[str, list[Piece]]]) -> list[Piece]:
    """One list of pieces out of the per-view detections of a scan (the overlaps merged)."""
    merged: list[Piece] = []
    for view, pieces in sightings:
        for piece in pieces:
            if piece.crop is not None and piece.crop.size and not piece.sighting_crops:
                piece.sighting_crops = [piece.crop]
            for known in merged:
                if ((known.x - piece.x) ** 2 + (known.y - piece.y) ** 2) ** 0.5 <= MERGE_RADIUS:
                    crops = known.sighting_crops + piece.sighting_crops
                    if piece.area > known.area:  # keep the better sighting, remember both views
                        piece.views = known.views + [view]
                        piece.sighting_crops = crops
                        merged[merged.index(known)] = piece
                    else:
                        known.views.append(view)
                        known.sighting_crops = crops
                    break
            else:
                piece.views = [view]
                merged.append(piece)
    merged.sort(key=lambda p: -p.area)
    return merged


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


LIGHT_CHANGE_MEDIAN = 15.0  # median difference over the board above which the baseline is stale


def light_change(registration: Any, frame: np.ndarray, baseline: Baseline) -> float:
    """Median difference between the rectified frame and the baseline where both saw the board.

    Pieces cover a small part of the board, so the median measures the light, not the game:
    about 2 with the light of the baseline, 28 the morning after an evening baseline.
    """
    width = baseline.image.shape[1]
    rectified = registration.rectify(frame, width=width)
    covered = registration.rectify(np.full(frame.shape[:2], 255, dtype=np.uint8), width=width)
    diff = difference_map(rectified, baseline)
    both = (covered == 255) & (baseline.coverage == 255)
    return float(np.median(diff[both])) if both.any() else 0.0


def adaptive_threshold(diff: np.ndarray, valid: np.ndarray) -> int:
    """Difference threshold from the frame's own noise: pieces cover little of the board, so the
    95th percentile of the difference measures the light and the sensor, not the game."""
    if not valid.any():
        return DIFF_THRESHOLD
    p95 = float(np.percentile(diff[valid], 95))
    return int(round(min(DIFF_THRESHOLD, max(DIFF_THRESHOLD_MIN, NOISE_FACTOR * p95))))


def find_pieces(
    registration: Registration,
    frame: np.ndarray,
    baseline: Baseline,
    *,
    threshold: int | None = None,
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
    both = (covered == 255) & (baseline.coverage == 255)  # only where both pictures saw the board
    if threshold is None:
        threshold = adaptive_threshold(diff, both)
    mask = (diff >= threshold).astype(np.uint8) * 255
    mask[~both] = 0
    mask[: int(h * BOARD_MARGIN_TOP), :] = 0
    mask[h - int(h * BOARD_MARGIN_BOTTOM) :, :] = 0
    for x0, y0, x1, y1 in (RESERVE_BOX, LEGEND_BOX):
        mask[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)] = 0
    fw, fh = registration.frame_size
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
        if area < min_area or area > max_area or min(bw, bh) < MIN_SIDE:
            continue
        cx, cy = (float(v) for v in centroids[i])
        strength = float(diff[labels == i].mean())
        # A standing piece is smeared away from the camera; the point where it touches the
        # board is its lowest point in the camera frame (nearest to the camera). Work on the
        # blob's contour in frame pixels: the direction "down" in the frame is reliable, a
        # direction computed on the map from far-away frame pixels is not (the horizon).
        contours, _ = cv2.findContours(
            (labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        outline = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32) * [scale_x, scale_y]
        in_frame = registration.to_frame(outline)
        lowest = in_frame[:, 1] >= np.percentile(in_frame[:, 1], 92)
        fbx, fby = (float(v) for v in in_frame[lowest].mean(axis=0))
        # A standee's lower part often matches the dark art it stands on and drops out of the
        # blob, so a tall blob ends above its contact point: put the base a little further down.
        frame_h = float(in_frame[:, 1].max() - in_frame[:, 1].min())
        frame_w = float(in_frame[:, 0].max() - in_frame[:, 0].min())
        if frame_w > 0 and frame_h / frame_w >= STANDEE_ASPECT:
            fby = min(fh - 1.0, fby + STANDEE_DROP * frame_h)
        base_ref = registration.to_reference([(fbx, fby)])[0]
        base_x, base_y = (
            float(base_ref[0]) / registration.reference_size[0],
            float(base_ref[1]) / registration.reference_size[1],
        )
        space, distance = nearest_space(base_x, base_y)
        near = None
        if space is None:
            candidate, distance = nearest_space(base_x, base_y, max_radius_factor=NEAR_FACTOR)
            near = candidate.name if candidate else None
        anchor = space or nearest_space(base_x, base_y, max_radius_factor=1e9)[0]
        radius_ratio = distance / anchor.radius if anchor else 0.0
        edge = not (
            EDGE_FRACTION * fw <= fbx <= (1 - EDGE_FRACTION) * fw
            and EDGE_FRACTION * fh <= fby <= (1 - EDGE_FRACTION) * fh
        )
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
                near=near,
                frame_box=box,
                crop=crop,
                frame_base=(float(fbx), float(fby)),
                radius_ratio=float(radius_ratio),
                edge=bool(edge),
                threshold=float(threshold),
            )
        )
    pieces.sort(key=lambda p: -p.area)
    return pieces


def closest_to(pieces: list[Piece], x: float, y: float, max_distance: float = 120.0) -> Piece | None:
    """The piece whose map position is nearest to (x, y), within ``max_distance`` map pixels."""
    best, best_d = None, max_distance
    for piece in pieces:
        d = ((piece.x - x) ** 2 + (piece.y - y) ** 2) ** 0.5
        if d <= best_d:
            best, best_d = piece, d
    return best


def classify_pieces(pieces: list[Piece]) -> None:
    """Dice among the pieces, with the value voted across every look at them."""
    from collections import Counter

    from src.vision.dice import read_die

    for piece in pieces:
        crops = [c for c in piece.sighting_crops if c is not None and c.size] or (
            [piece.crop] if piece.crop is not None and piece.crop.size else []
        )
        readings = [read_die(c) for c in crops]
        dice = [r for r in readings if r.is_die]
        if not readings or len(dice) * 2 < len(readings):
            continue
        piece.kind = "die"
        values = Counter(r.value for r in dice if r.value is not None)
        if not values:
            piece.value, piece.kind_confidence = None, 0.2
            continue
        value, votes = values.most_common(1)[0]
        piece.value = value
        piece.kind_confidence = round(votes / len(dice), 2) if len(dice) > 1 else 0.5


def describe(pieces: list[Piece]) -> str:
    """One English sentence per piece, for the log and for the voice later."""
    if not pieces:
        return "Nothing on the board that is not the board."
    parts = []
    for piece in pieces:
        if piece.space:
            where = f"at {piece.space}"
        elif piece.near:
            where = f"near {piece.near}, not on it"
        else:
            where = "between spaces"
        note = ", confirmed by a closer look" if piece.confirmed else ""
        if piece.kind == "die":
            shown = f"showing {piece.value}" if piece.value is not None else "value unread"
            sure = "" if piece.kind_confidence >= 0.6 else " (unsure)"
            parts.append(f"a die {shown}{sure} {where}{note}")
        else:
            parts.append(
                f"something {where} ({int(piece.width)}x{int(piece.height)} px, strength {piece.strength:.0f}{note})"
            )
    return "; ".join(parts) + "."


__all__ = [
    "Piece",
    "Baseline",
    "BaselineSet",
    "difference_map",
    "find_pieces",
    "adaptive_threshold",
    "light_change",
    "merge_pieces",
    "classify_pieces",
    "closest_to",
    "describe",
    "MAP_WIDTH",
]
