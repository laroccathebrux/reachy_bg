"""Dice on the board: is this piece a die, and what does it show?

    verdict = read_die(crop)        # DieReading(is_die=True, value=5, confidence=0.7) or is_die=False

The game's dice are dark cubes with light pips. In a crop of the frame around a piece the die
body is the largest dark blob; it must be small and roughly square. Pips are light, round
blobs of a size proportional to the body, counted inside the central part of the body so
that the pips of the side faces (visible at an angle) and the lit side face itself are left
out. The count is the value; anything outside 1-6 lowers the confidence. At this camera's
distance a die is about 60 pixels across, so the reading is a best effort: the game loop
should confirm a low-confidence value by voice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.logger import get_logger

log = get_logger(__name__)

UPSCALE = 4
BODY_GREY_MAX = 70  # a die body is darker than this (0..255): black plastic, not the board's dark art
MIN_BODY_PX = 25  # in original frame pixels: smaller dark blobs are shadows or print
MAX_BODY_PX = 140
INNER_FRACTION = 0.22  # of the body size eroded on every side before counting pips
PIP_MIN_AREA_FRACTION = 0.004  # of the body area (at the upscaled size)
PIP_MAX_AREA_FRACTION = 0.06


@dataclass(frozen=True)
class DieReading:
    is_die: bool
    value: int | None = None
    confidence: float = 0.0
    body_px: int = 0
    blobs: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "is_die": self.is_die,
            "value": self.value,
            "confidence": round(self.confidence, 2),
            "body_px": self.body_px,
        }


def _largest_dark_body(grey: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """The dark blob nearest to the crop's centre (the crop is centred on the piece), big enough."""
    import cv2

    dark = (grey < BODY_GREY_MAX).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    if count < 2:
        return None, (0, 0, 0, 0)
    ch, cw = grey.shape[0] / 2, grey.shape[1] / 2
    smallest = (MIN_BODY_PX * UPSCALE) ** 2 * 0.4
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= smallest]
    if not candidates:
        return None, (0, 0, 0, 0)
    index = min(candidates, key=lambda i: (centroids[i][0] - cw) ** 2 + (centroids[i][1] - ch) ** 2)
    x, y, w, h, _ = (int(v) for v in stats[index])
    mask = (labels == index).astype(np.uint8) * 255
    return mask, (x, y, w, h)


def read_die(crop: np.ndarray) -> DieReading:
    """Best-effort reading of a die in ``crop`` (BGR, original frame resolution)."""
    import cv2

    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
        return DieReading(False)
    big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    grey = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    mask, (x, y, w, h) = _largest_dark_body(grey)
    if mask is None:
        return DieReading(False)
    size = max(w, h) / UPSCALE
    aspect = w / max(1, h)
    if not (MIN_BODY_PX <= size <= MAX_BODY_PX and 0.6 <= aspect <= 1.7):
        return DieReading(False, body_px=int(size))
    hull = cv2.convexHull(np.argwhere(mask > 0)[:, ::-1].astype(np.int32))
    full = np.zeros_like(grey)
    cv2.fillConvexPoly(full, hull, 255)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, int(w * INNER_FRACTION)) | 1, max(3, int(h * INNER_FRACTION)) | 1)
    )
    inner = cv2.erode(full, kernel)
    body_area = float(cv2.countNonZero(full))
    inside = grey[full > 0]
    floor = float(np.median(inside))
    # Pips are the light, round spots inside the die: local threshold above the body's own tone.
    threshold = floor + max(30.0, 0.5 * (float(np.percentile(inside, 99)) - floor))
    light = ((grey > threshold) & (inner > 0)).astype(np.uint8) * 255
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(light)
    pips = 0
    for i in range(1, count):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if not (PIP_MIN_AREA_FRACTION * body_area <= area <= PIP_MAX_AREA_FRACTION * body_area):
            continue
        if not (0.5 <= bw / max(1, bh) <= 2.0):
            continue
        if area < 0.45 * bw * bh:  # not filled enough to be a round spot
            continue
        pips += 1
    if pips == 0:
        return DieReading(True, None, 0.2, int(size), 0)
    value = min(6, pips)
    confidence = 0.75 if 1 <= pips <= 6 else 0.3
    return DieReading(True, value, confidence, int(size), pips)


def describe_die(reading: DieReading) -> str:
    if not reading.is_die:
        return "not a die"
    if reading.value is None:
        return "a die, value unread"
    return f"a die showing {reading.value}" + ("" if reading.confidence >= 0.6 else " (unsure)")


__all__ = ["DieReading", "read_die", "describe_die"]
