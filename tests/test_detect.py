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


def test_a_piece_next_to_a_space_is_reported_as_near_it():
    from src.vision.detect import Piece, describe

    piece = Piece(10, 10, 5, 5, 400, 70.0, None, 0.06, near="Rome")
    assert "near Rome" in describe([piece]) and piece.record()["near"] == "Rome"


def test_merge_pieces_joins_sightings_of_the_same_spot():
    from src.vision.detect import BaselineSet, Piece, merge_pieces

    a = Piece(100, 100, 20, 20, 400, 60.0, "Rome", 0.01)
    b = Piece(110, 105, 30, 30, 900, 62.0, "Rome", 0.02)  # the same piece, a better look at it
    c = Piece(600, 400, 20, 20, 300, 55.0, "Tokyo", 0.01)
    merged = merge_pieces([("centre", [a]), ("left", [b, c])])
    assert [(p.space, sorted(p.views), p.area) for p in merged] == [
        ("Rome", ["centre", "left"], 900),
        ("Tokyo", ["left"], 300),
    ]
    assert merge_pieces([]) == []
    assert BaselineSet().views == {}


def test_pieces_on_the_reserve_and_the_legend_are_ignored():
    board = synthetic_board()
    reference = BoardReference(image=board, min_inliers=20)
    empty, true_h = perspective_frame(board)
    registration = reference.locate(empty)
    baseline = Baseline.capture(registration, empty)
    busy = board.copy()
    h, w = board.shape[:2]
    cv2.rectangle(
        busy, (int(w * 0.1), int(h * 0.88)), (int(w * 0.18), int(h * 0.97)), (30, 30, 230), -1
    )  # Reserve
    cv2.circle(busy, (int(w * 0.08), int(h * 0.6)), 14, (30, 30, 230), -1)  # legend
    frame = cv2.warpPerspective(busy, true_h, (empty.shape[1], empty.shape[0]))
    registration = reference.locate(frame)
    assert find_pieces(registration, frame, baseline, min_area=80) == []


def test_needs_closer_look_rules():
    from src.vision.detect import Piece, closest_to

    strong = Piece(100, 100, 20, 20, 400, 80.0, "Rome", 0.01, radius_ratio=0.2)
    weak = Piece(100, 100, 20, 20, 400, 50.0, "Rome", 0.01, radius_ratio=0.2)
    off_centre = Piece(100, 100, 20, 20, 400, 80.0, "Rome", 0.03, radius_ratio=0.8)
    at_edge = Piece(100, 100, 20, 20, 400, 80.0, "Rome", 0.01, radius_ratio=0.2, edge=True)
    between = Piece(100, 100, 20, 20, 400, 80.0, None, 0.06, near="Rome", radius_ratio=1.5)
    assert not strong.needs_closer_look()
    assert all(p.needs_closer_look() for p in (weak, off_centre, at_edge, between))
    strong.confirmed = True
    assert not strong.needs_closer_look()
    assert closest_to([strong, Piece(500, 500, 9, 9, 100, 70.0, "Tokyo", 0.0)], 110, 95) is strong
    assert closest_to([strong], 400, 400) is None
