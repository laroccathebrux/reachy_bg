"""Board registration on a synthetic textured board warped by a known homography (no camera, no real board)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.board_map import BoardReference, draw_registration  # noqa: E402
from src.vision.spaces import BY_NAME, SPACES, nearest_space  # noqa: E402


def synthetic_board(width=800, height=520, seed=3):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    board = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(60):  # some sharp shapes so SIFT has corners to work with
        x, y = int(rng.integers(20, width - 20)), int(rng.integers(20, height - 20))
        cv2.rectangle(
            board, (x, y), (x + int(rng.integers(8, 30)), y + int(rng.integers(8, 30))), (255, 255, 255), -1
        )
        cv2.circle(board, (x, y), int(rng.integers(4, 12)), (0, 0, 0), -1)
    return board


def perspective_frame(board, frame_w=640, frame_h=400):
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [[-60, 40], [700, 20], [560, 390], [90, 380]]
    )  # the board seen at an angle, cut at the sides
    true_h = cv2.getPerspectiveTransform(src, dst)  # board -> frame
    frame = cv2.warpPerspective(board, true_h, (frame_w, frame_h))
    return frame, true_h


def test_locate_recovers_the_homography_and_projects_spaces():
    board = synthetic_board()
    reference = BoardReference(image=board, min_inliers=25)
    frame, true_h = perspective_frame(board)
    registration = reference.locate(frame)
    assert registration is not None and registration.inliers >= 25
    # A point of the board -> frame by the truth -> back to the board by the registration.
    london = BY_NAME["London"].pixel(reference.width, reference.height)
    in_frame = cv2.perspectiveTransform(np.float32([[london]]), true_h).reshape(2)
    back = registration.to_reference([in_frame])[0]
    assert np.allclose(back, london, atol=3.0)
    space, distance = registration.space_at(float(in_frame[0]), float(in_frame[1]))
    assert space is not None and space.name == "London"
    pixels = registration.space_pixels()
    assert "London" in pixels and np.allclose(pixels["London"], in_frame, atol=3.0)
    assert registration.outline().shape == (4, 2)
    top_down = registration.rectify(frame, width=400)
    assert top_down.shape[:2] == (260, 400)
    drawn = draw_registration(frame, registration)
    assert drawn.shape == frame.shape and not np.array_equal(drawn, frame)


def test_locate_rejects_a_frame_without_the_board():
    reference = BoardReference(image=synthetic_board(), min_inliers=25)
    other = synthetic_board(seed=99)
    assert reference.locate(np.zeros((300, 400, 3), dtype=np.uint8)) is None
    assert reference.locate(other) is None


def test_locate_on_a_downscaled_copy_gives_the_same_geometry():
    board = synthetic_board()
    reference = BoardReference(image=board, min_inliers=20)
    frame, _ = perspective_frame(board)
    full = reference.locate(frame)
    half = reference.locate(frame, scale=0.5)
    assert full is not None and half is not None
    corner_full = full.to_reference([(320, 200)])[0]
    corner_half = half.to_reference([(320, 200)])[0]
    assert np.allclose(corner_full, corner_half, atol=6.0)


def test_spaces_table_and_nearest():
    names = [s.name for s in SPACES]
    assert len(names) == len(set(names)) == 36  # 9 cities, 6 expedition sites, 21 numbered
    assert {s.kind for s in SPACES} == {"city", "wilderness", "sea"}
    london = BY_NAME["London"]
    assert nearest_space(london.x + 0.01, london.y)[0] is london
    assert nearest_space(0.5, 0.95)[0] is None  # the Reserve strip: not a space
