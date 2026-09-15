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
INNER_FRACTION = 0.12  # of the body size eroded on every side before counting pips
# Seen from the robot's head (about 35 degrees above the table) a die shows its top face,
# compressed to roughly the upper 45 % of its outline, and its front face below it, larger
# and with rounder pips. Only the spots above this fraction of the body height are the value.
TOP_FACE_FRACTION = 0.55
TOP_FACE_BAND = 0.40  # between this and TOP_FACE_FRACTION a spot must look like a compressed top pip
# A flat sliver on the top edge looked like a glint but was a pip cut by the edge (four known dice:
# 4/4 right by vote without this filter, 3/4 with it); the filter stays off.
EDGE_GLINT_ASPECT = 99.0
GAP_SEARCH_MIN, GAP_SEARCH_MAX = 0.28, 0.72  # of the body height: where the edge between faces can be
FRONT_PEEK_BAND = 0.14  # of the body height above the edge where a front pip can still appear
MIN_FACE_GAP = 0.10  # of the body height: smaller gaps are just the spacing of pips within a face
MIN_SPOTS = 2  # a die seen from above at an angle shows two faces: one lone spot is a highlight
MERGED_PIP_AREA = 1.6  # a spot this many times the median pip area is two pips touching
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


def _top_face_limit(rows: list[float]) -> float:
    """Where the top face ends: the largest vertical gap between spot rows, else a fixed fraction.

    The pips of the top face and of the front face form two groups separated by a gap (the
    edge between the faces); the gap is looked for in the middle of the body so that a gap
    inside one face's own rows does not split it.
    """
    best_gap, limit = 0.0, TOP_FACE_FRACTION
    for above, below in zip(rows, rows[1:], strict=False):
        split = (above + below) / 2
        gap = below - above
        if GAP_SEARCH_MIN <= split <= GAP_SEARCH_MAX and gap >= MIN_FACE_GAP and gap > best_gap:
            best_gap, limit = gap, split
    return limit


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
    count, _, stats, centroids = cv2.connectedComponentsWithStats(light)
    spots: list[tuple[float, float, float, float]] = []  # x, y, area, width over height
    for i in range(1, count):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if not (PIP_MIN_AREA_FRACTION * body_area <= area <= PIP_MAX_AREA_FRACTION * body_area):
            continue
        if not (0.4 <= bw / max(1, bh) <= 2.5):
            continue
        spots.append((float(centroids[i][0]), float(centroids[i][1]), float(area), bw / max(1, bh)))
    if spots:  # highlights are much smaller than the pips
        median_area = float(np.median([a for _, _, a, _ in spots]))
        spots = [sp for sp in spots if sp[2] >= 0.4 * median_area]
    if len(spots) < MIN_SPOTS:  # a black block with at most a highlight: a standee's plastic base
        return DieReading(False, None, 0.0, int(size), len(spots))
    median_area = float(np.median([a for _, _, a, _ in spots]))
    # Highlights on the top edge are flat slivers, smaller than a pip.
    spots = [sp for sp in spots if not (sp[3] > EDGE_GLINT_ASPECT and sp[2] < 0.75 * median_area)]
    if len(spots) < MIN_SPOTS:
        return DieReading(False, None, 0.0, int(size), len(spots))
    rows = sorted((sy - y) / max(1, h) for _, sy, _, _ in spots)
    limit = _top_face_limit(rows)
    pips = 0
    for _, sy, area, aspect in spots:
        fy = (sy - y) / max(1, h)
        if fy >= limit:
            continue  # front face
        if fy >= limit - FRONT_PEEK_BAND and (aspect < 0.95 or area < 0.75 * median_area):
            continue  # the front face's top pip peeking above the edge: taller than wide, smaller
        pips += 2 if area >= MERGED_PIP_AREA * median_area else 1
    if pips == 0:
        return DieReading(False, None, 0.0, int(size), len(spots))
    value = min(6, pips)
    confidence = 0.7 if pips <= 6 else 0.3
    return DieReading(True, value, confidence, int(size), len(spots))


def describe_die(reading: DieReading) -> str:
    if not reading.is_die:
        return "not a die"
    if reading.value is None:
        return "a die, value unread"
    return f"a die showing {reading.value}" + ("" if reading.confidence >= 0.6 else " (unsure)")


__all__ = ["DieReading", "read_die", "describe_die"]
