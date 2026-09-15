"""Reserve slots on a synthetic board: a card drawn in slot 2 is the only occupied slot."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.board_map import BoardReference  # noqa: E402
from src.vision.detect import Baseline  # noqa: E402
from src.vision.reserve import RESERVE_SLOTS, describe_reserve, read_reserve  # noqa: E402
from tests.test_board_map import synthetic_board  # noqa: E402


def reserve_frame(board, frame_w=640, frame_h=400):
    """A view that covers the whole board (the Reserve included) at a mild angle."""
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 30], [610, 20], [630, 390], [10, 380]])
    true_h = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, true_h, (frame_w, frame_h)), true_h


def test_card_in_slot_two():
    board = synthetic_board()
    reference = BoardReference(image=board, min_inliers=20)
    empty, true_h = reserve_frame(board)
    registration = reference.locate(empty)
    assert registration is not None
    baseline = Baseline.capture(registration, empty)
    busy = board.copy()
    h, w = board.shape[:2]
    x0, y0, x1, y1 = RESERVE_SLOTS[1]
    cv2.rectangle(
        busy, (int(x0 * w) + 2, int(y0 * h) + 2), (int(x1 * w) - 2, int(y1 * h) - 2), (40, 200, 240), -1
    )
    frame = cv2.warpPerspective(busy, true_h, (empty.shape[1], empty.shape[0]))
    registration = reference.locate(frame)
    slots = read_reserve(registration, frame, baseline)
    assert [s.occupied for s in slots] == [False, True, False, False]
    assert all(s.seen for s in slots) and slots[1].fraction > 0.8 and slots[0].fraction < 0.1
    assert slots[1].crop is not None and slots[1].crop.size > 0
    assert describe_reserve(slots) == "Reserve: 1 card in slot 2."
    assert slots[1].record()["slot"] == 2
    assert describe_reserve(read_reserve(registration, empty, baseline)) == "Reserve: 0 cards."
