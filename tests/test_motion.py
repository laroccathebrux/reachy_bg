"""Reading a move on a synthetic board: a hand passes over, a piece ends up somewhere else."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.board_map import BoardReference  # noqa: E402
from src.vision.motion import MotionWatcher, Move, activity, compare  # noqa: E402
from src.vision.spaces import BY_NAME  # noqa: E402
from tests.test_board_map import perspective_frame, synthetic_board  # noqa: E402

MIN_AREA = 80  # the synthetic token is smaller than a real standee


def board_with_token(board, reference, space_name, true_h, size):
    """The board seen through the same perspective, with a red disc on ``space_name``."""
    x, y = BY_NAME[space_name].pixel(reference.width, reference.height)
    painted = board.copy()
    cv2.circle(painted, (int(x), int(y)), 14, (30, 30, 230), -1)
    return cv2.warpPerspective(painted, true_h, size)


def a_hand_over(frame):
    """A big dark shape across the middle of the frame, like an arm reaching in."""
    h, w = frame.shape[:2]
    out = frame.copy()
    cv2.rectangle(out, (0, h // 3), (w, 2 * h // 3), (20, 20, 20), -1)
    return out


@pytest.fixture
def view():
    board = synthetic_board()
    reference = BoardReference(image=board, min_inliers=20)
    empty, true_h = perspective_frame(board)
    registration = reference.locate(empty)
    assert registration is not None
    size = (empty.shape[1], empty.shape[0])
    return board, reference, registration, empty, true_h, size


def test_activity_is_near_zero_between_identical_frames(view):
    _, _, _, empty, _, _ = view
    assert activity(empty, empty) == 0.0


def test_activity_is_high_when_a_hand_covers_the_board(view):
    _, _, _, empty, _, _ = view
    assert activity(empty, a_hand_over(empty)) > 0.1


def test_compare_reports_a_token_that_arrived(view):
    board, reference, registration, empty, true_h, size = view
    after = board_with_token(board, reference, "London", true_h, size)
    move = compare(registration, empty, after, min_area=MIN_AREA)
    assert not move.empty
    assert [p.space for p in move.arrived] == ["London"]
    assert move.departed == []
    assert "London" in move.describe()


def test_compare_reports_a_token_that_left(view):
    board, reference, registration, empty, true_h, size = view
    before = board_with_token(board, reference, "London", true_h, size)
    move = compare(registration, before, empty, min_area=MIN_AREA)
    assert [p.space for p in move.departed] == ["London"]
    assert move.arrived == []


def test_compare_reads_a_move_from_one_space_to_another(view):
    board, reference, registration, _, true_h, size = view
    before = board_with_token(board, reference, "London", true_h, size)
    after = board_with_token(board, reference, "Rome", true_h, size)
    move = compare(registration, before, after, min_area=MIN_AREA)
    assert [p.space for p in move.departed] == ["London"]
    assert [p.space for p in move.arrived] == ["Rome"]
    assert move.describe() == "a piece moved from London to Rome"
    assert move.record()["text"] == move.describe()


def test_a_still_board_never_reports_a_move(view):
    _, _, registration, empty, _, _ = view
    watcher = MotionWatcher(registration, min_area=MIN_AREA)
    for _ in range(20):
        assert watcher.feed(empty) is None
    assert watcher.state == "quiet" and watcher.moves == []


def test_the_watcher_reports_the_move_once_the_board_settles(view):
    board, reference, registration, empty, true_h, size = view
    after = board_with_token(board, reference, "London", true_h, size)
    watcher = MotionWatcher(registration, settle_frames=3, min_area=MIN_AREA)

    for _ in range(4):  # the board at rest: the anchor tracks the empty board
        assert watcher.feed(empty) is None
    for _ in range(3):  # a hand reaches over it
        assert watcher.feed(a_hand_over(empty)) is None
    assert watcher.state == "disturbed"

    moves = [watcher.feed(after) for _ in range(4)]  # the hand leaves, a token is now on London
    reported = [m for m in moves if m is not None]
    assert len(reported) == 1, f"expected exactly one verdict, got {len(reported)}"
    assert [p.space for p in reported[0].arrived] == ["London"]
    assert watcher.state == "quiet" and reported[0].seconds >= 0.0


def test_the_watcher_waits_for_the_board_to_be_still_again(view):
    _, _, registration, empty, _, _ = view
    watcher = MotionWatcher(registration, settle_frames=5, min_area=MIN_AREA)
    watcher.feed(empty)
    watcher.feed(a_hand_over(empty))
    assert watcher.state == "disturbed"
    for _ in range(4):  # four quiet frames are not yet the five it asks for
        assert watcher.feed(empty) is None
    assert watcher.state == "disturbed"


def test_reset_forgets_the_history(view):
    _, _, registration, empty, _, _ = view
    watcher = MotionWatcher(registration, min_area=MIN_AREA)
    watcher.feed(empty)
    watcher.feed(a_hand_over(empty))
    assert watcher.state == "disturbed"
    watcher.reset(empty)
    assert watcher.state == "quiet" and watcher.quiet_run == 0


def test_an_empty_move_describes_itself():
    move = Move()
    assert move.empty and move.describe() == "nothing changed"
    assert move.record()["departed"] == []


def test_a_move_off_the_spaces_is_still_described():
    from src.vision.detect import Piece

    move = Move(arrived=[Piece(10, 10, 5, 5, 400, 70.0, None, 0.06, near="Rome")])
    assert "near Rome" in move.describe()
    move = Move(arrived=[Piece(10, 10, 5, 5, 400, 70.0, None, 0.9)])
    assert "off the spaces" in move.describe()


def test_activity_of_frames_of_different_sizes_is_total(view):
    _, _, _, empty, _, _ = view
    assert activity(empty, np.zeros((10, 10, 3), dtype=np.uint8)) == 1.0
