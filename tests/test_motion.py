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


def mjpeg_bytes(images, boundary=b"--reachyframe"):
    """A multipart MJPEG body, the way the preview serves one."""
    out = b""
    for img in images:
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        payload = buf.tobytes()
        out += boundary + b"\r\nContent-Type: image/jpeg\r\n"
        out += b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
        out += payload + b"\r\n"
    return out


def test_iter_jpegs_splits_a_stream_into_frames():
    import io

    from src.vision.motion import iter_jpegs

    images = [np.full((16, 24, 3), v, dtype=np.uint8) for v in (10, 120, 240)]
    payloads = list(iter_jpegs(io.BytesIO(mjpeg_bytes(images)), chunk=7))
    assert len(payloads) == 3
    for payload, original in zip(payloads, images, strict=True):
        assert payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == original.shape
        assert abs(int(decoded.mean()) - int(original.mean())) <= 2


def test_iter_jpegs_stops_at_the_end_of_the_stream():
    import io

    from src.vision.motion import iter_jpegs

    assert list(iter_jpegs(io.BytesIO(b""), chunk=4)) == []
    assert list(iter_jpegs(io.BytesIO(b"\xff\xd8 no end marker"), chunk=4)) == []


def test_the_watcher_ignores_a_repeated_frame(view):
    """A source that keeps only the newest frame hands the same one out between captures."""
    board, reference, registration, empty, true_h, size = view
    after = board_with_token(board, reference, "London", true_h, size)

    class Repeating:
        """Serves each frame several times, counting only the real captures."""

        def __init__(self, script):
            self.script = script
            self.frames = 0
            self.i = -1

        def get_frame(self):
            return self.script[min(self.i, len(self.script) - 1)] if self.i >= 0 else None

        def advance(self):
            self.i += 1
            self.frames += 1

    from src.vision.motion import MotionWatcher

    watcher = MotionWatcher(registration, settle_frames=3, min_area=MIN_AREA)
    camera = Repeating([empty] * 3 + [a_hand_over(empty)] * 2 + [after] * 4)
    moves, last_count = [], -1
    for _ in range(40):  # poll faster than the camera produces, as watch() does
        if camera.frames == last_count:
            camera.advance()
            continue
        last_count = camera.frames
        frame = camera.get_frame()
        if frame is None:
            camera.advance()
            continue
        move = watcher.feed(frame)
        if move:
            moves.append(move)
        camera.advance()
    assert len(moves) == 1, f"expected one verdict, got {len(moves)}"
    assert [p.space for p in moves[0].arrived] == ["London"]


def test_a_turned_view_is_refused_instead_of_read_as_a_move(view):
    """A body sweep once produced "12 pieces left": the registration belonged to the old view."""
    board, reference, registration, empty, true_h, _ = view
    watcher = MotionWatcher(registration, settle_frames=2, min_area=MIN_AREA, reference=reference)

    # the same board from a different camera pose, the way a body sweep leaves it
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[10, 30], [740, 50], [620, 380], [150, 370]])
    turned = cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (640, 400))
    assert watcher.view_shifted(turned), "the moved view was not noticed"
    assert not watcher.view_shifted(empty), "the unchanged view was called stale"

    watcher.feed(empty)
    watcher.feed(a_hand_over(empty))
    assert watcher.state == "disturbed"
    verdicts = [watcher.feed(turned) for _ in range(5)]
    assert all(v is None for v in verdicts), "a verdict was read from a stale registration"
    assert watcher.moves == []


def test_a_verdict_naming_too_many_pieces_is_dropped(view):
    _, _, registration, empty, _, _ = view
    watcher = MotionWatcher(registration, settle_frames=2, min_area=MIN_AREA, max_pieces=1)

    class ManyPieces:
        pass

    from src.vision import motion as motion_module
    from src.vision.detect import Piece

    original = motion_module.compare
    motion_module.compare = lambda *a, **k: Move(
        departed=[Piece(10, 10, 5, 5, 400, 70.0, "London", 0.0)],
        arrived=[Piece(20, 20, 5, 5, 400, 70.0, "Rome", 0.0), Piece(30, 30, 5, 5, 400, 70.0, "Tokyo", 0.0)],
    )
    try:
        watcher.feed(empty)
        watcher.feed(a_hand_over(empty))
        verdicts = [watcher.feed(empty) for _ in range(4)]
        assert all(v is None for v in verdicts), "a 3-piece verdict passed a max_pieces of 1"
        assert watcher.moves == []
    finally:
        motion_module.compare = original


def test_the_baseline_decides_direction_over_busy_board_art(view):
    """The live failure: a piece landing on a dark card was called a departure.

    ``stands_out`` asks how far the blob sits from its surroundings, which barely moves when a
    piece lands on art that is already dark. ``occupancy`` asks whether the spot differs from
    the *empty* board, which is unambiguous.
    """
    from src.vision.detect import Baseline
    from src.vision.motion import occupancy

    board, reference, registration, empty, true_h, size = view
    baseline = Baseline.capture(registration, empty)

    # A dark card printed on the board at Rome, with the piece landing on top of it later.
    with_card = board.copy()
    cx, cy = BY_NAME["Rome"].pixel(reference.width, reference.height)
    cv2.rectangle(with_card, (int(cx) - 22, int(cy) - 22), (int(cx) + 22, int(cy) + 22), (35, 30, 25), -1)
    before = cv2.warpPerspective(with_card, true_h, size)
    on_the_card = with_card.copy()
    cv2.circle(on_the_card, (int(cx), int(cy)), 13, (30, 30, 230), -1)
    after = cv2.warpPerspective(on_the_card, true_h, size)

    base_of_card = Baseline.capture(registration, before)  # the board *with* its printed card
    move = compare(registration, before, after, min_area=MIN_AREA, baseline=base_of_card)
    assert [p.space for p in move.arrived] == ["Rome"], f"got {move.describe()}"
    assert move.departed == []

    rect_after = registration.rectify(after, width=base_of_card.image.shape[1])
    piece = move.arrived[0]
    assert occupancy(rect_after, base_of_card, piece) > 0.0
    assert occupancy(rect_after, baseline, piece) > 0.0
