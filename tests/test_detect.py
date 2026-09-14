"""Foreground pieces on a synthetic board: baseline of the empty board, then a token placed on a space."""

import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.board_map import BoardReference  # noqa: E402
from src.vision.detect import Baseline, describe, find_pieces  # noqa: E402
from src.vision.spaces import BY_NAME  # noqa: E402
from tests.test_board_map import perspective_frame, synthetic_board  # noqa: E402


def test_token_on_london_is_found_and_named(tmp_path):
    board = synthetic_board()
    reference = BoardReference(image=board, min_inliers=20)
    empty_frame, true_h = perspective_frame(board)
    registration = reference.locate(empty_frame)
    assert registration is not None
    baseline = Baseline.capture(registration, empty_frame)
    assert baseline.image.shape[1] == 1200 and baseline.coverage.max() == 255
    baseline.save(tmp_path / "baseline.jpg")
    baseline = Baseline.load(tmp_path / "baseline.jpg")

    # Put a bright red disc on London (board pixels), then look at the board again.
    london = BY_NAME["London"].pixel(reference.width, reference.height)
    with_token = board.copy()
    cv2.circle(with_token, (int(london[0]), int(london[1])), 14, (30, 30, 230), -1)
    frame = cv2.warpPerspective(with_token, true_h, (empty_frame.shape[1], empty_frame.shape[0]))
    registration = reference.locate(frame)
    assert registration is not None
    pieces = find_pieces(registration, frame, baseline, min_area=80)
    assert pieces, "the token was not found"
    assert pieces[0].space == "London" and pieces[0].crop is not None and pieces[0].crop.size > 0
    x0, y0, x1, y1 = pieces[0].frame_box
    assert 0 <= x0 < x1 <= frame.shape[1] and 0 <= y0 < y1 <= frame.shape[0]
    assert "London" in describe(pieces)
    assert pieces[0].record()["space"] == "London"


def test_empty_board_has_no_pieces():
    board = synthetic_board()
    reference = BoardReference(image=board, min_inliers=20)
    frame, _ = perspective_frame(board)
    registration = reference.locate(frame)
    baseline = Baseline.capture(registration, frame)
    assert find_pieces(registration, frame, baseline) == []
    assert describe([]).startswith("Nothing")
