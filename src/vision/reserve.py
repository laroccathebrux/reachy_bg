"""The Reserve: four card slots at the bottom-left of the board, read as occupied or empty.

    slots = read_reserve(registration, frame, baseline)      # -> [Slot(index=1, occupied=True, ...), ...]

A card in a slot covers the printed slot outline, so on the rectified map the difference
against the empty-board baseline fills most of the slot rectangle. Each occupied slot gets a
crop of the card at full camera resolution, the sample for naming the card later (a gallery
of the Asset cards photographed by this camera). The slots are only judged when both the
frame and the baseline actually covered them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.logger import get_logger
from src.vision.detect import Baseline, difference_map

log = get_logger(__name__)

# Slot rectangles in normalised map coordinates (measured on the 1200x790 reference picture).
RESERVE_SLOTS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x0 / 1200, 684 / 790, x1 / 1200, 790 / 790) for x0, x1 in ((94, 174), (182, 256), (262, 336), (344, 418))
)
SLOT_INSET = 0.12  # fraction of the slot shrunk on every side: the printed outline is not a card
SLOT_PICTURE_BOTTOM = 0.55  # the upper part of a card is its picture: strong against the cream slot;
# the text half is cream on cream and barely differs. Measured: cards 0.62-0.73 of the picture
# window above 20, empty slots 0.00 (another empty view against the baseline).
SLOT_THRESHOLD = 20
OCCUPIED_FRACTION = 0.3  # of the picture window differing from the baseline
COVERED_FRACTION = 0.8  # of the slot that both pictures must have seen to judge it


@dataclass
class Slot:
    index: int  # 1..4, left to right
    occupied: bool
    fraction: float  # of the slot area that differs from the empty board
    seen: bool  # both the frame and the baseline covered the slot
    frame_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    crop: np.ndarray | None = None

    def record(self) -> dict[str, Any]:
        return {
            "slot": self.index,
            "occupied": self.occupied,
            "fraction": round(self.fraction, 3),
            "seen": self.seen,
            "frame_box": list(self.frame_box),
        }


def read_reserve(
    registration: Any,
    frame: np.ndarray,
    baseline: Baseline,
    *,
    threshold: int = SLOT_THRESHOLD,
    occupied_fraction: float = OCCUPIED_FRACTION,
) -> list[Slot]:
    width = baseline.image.shape[1]
    rectified = registration.rectify(frame, width=width)
    covered = registration.rectify(np.full(frame.shape[:2], 255, dtype=np.uint8), width=width)
    diff = difference_map(rectified, baseline)
    h, w = diff.shape
    ref_w, ref_h = registration.reference_size
    fw, fh = registration.frame_size
    slots: list[Slot] = []
    for index, (x0, y0, x1, y1) in enumerate(RESERVE_SLOTS, start=1):
        ix0, ix1 = x0 + (x1 - x0) * SLOT_INSET, x1 - (x1 - x0) * SLOT_INSET
        iy0, iy1 = y0 + (y1 - y0) * SLOT_INSET, y0 + (y1 - y0) * SLOT_PICTURE_BOTTOM
        px0, px1 = int(ix0 * w), int(ix1 * w)
        py0, py1 = int(iy0 * h), min(int(iy1 * h), h - 1)
        window = diff[py0:py1, px0:px1]
        both = (covered[py0:py1, px0:px1] == 255) & (baseline.coverage[py0:py1, px0:px1] == 255)
        seen = window.size > 0 and float(both.mean()) >= COVERED_FRACTION
        fraction = float((window[both] >= threshold).mean()) if seen and both.any() else 0.0
        corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * [ref_w, ref_h]
        in_frame = registration.to_frame(corners)
        bx0, by0 = in_frame.min(axis=0)
        bx1, by1 = in_frame.max(axis=0)
        box = (int(max(0, bx0)), int(max(0, by0)), int(min(fw, bx1)), int(min(fh, by1)))
        crop = frame[box[1] : box[3], box[0] : box[2]].copy() if box[2] > box[0] and box[3] > box[1] else None
        slots.append(Slot(index, seen and fraction >= occupied_fraction, fraction, seen, box, crop))
    return slots


def describe_reserve(slots: list[Slot]) -> str:
    if not slots:
        return "Reserve not read."
    unseen = [s.index for s in slots if not s.seen]
    cards = [s.index for s in slots if s.occupied]
    text = f"Reserve: {len(cards)} card{'s' if len(cards) != 1 else ''}"
    if cards:
        text += " in slot" + ("s " if len(cards) > 1 else " ") + ", ".join(str(i) for i in cards)
    if unseen:
        text += f"; slot{'s' if len(unseen) > 1 else ''} {', '.join(str(i) for i in unseen)} not in view"
    return text + "."


__all__ = ["Slot", "RESERVE_SLOTS", "read_reserve", "describe_reserve"]
