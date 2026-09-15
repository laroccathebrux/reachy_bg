"""Synthetic dice: a dark square with white pips, blurred like the camera, read back."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.dice import describe_die, read_die  # noqa: E402

LAYOUTS = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
    6: [(0.25, 0.2), (0.75, 0.2), (0.25, 0.5), (0.75, 0.5), (0.25, 0.8), (0.75, 0.8)],
}


def synthetic_die(value, front=3, size=100, margin=30, blur=3):
    """A die as the robot sees it: the top face compressed into the upper 45 % of the outline, the
    front face (showing ``front``) below it, larger and with rounder pips."""
    canvas = np.full(
        (size + 2 * margin, size + 2 * margin, 3), (170, 190, 200), dtype=np.uint8
    )  # cream board
    cv2.rectangle(canvas, (margin, margin), (margin + size, margin + size), (25, 25, 25), -1)
    top_h = int(size * 0.45)
    for fx, fy in LAYOUTS[value]:
        centre = (int(margin + fx * size), int(margin + fy * top_h))
        cv2.ellipse(canvas, centre, (size // 14, size // 20), 0, 0, 360, (235, 235, 235), -1)  # same area as below
    for fx, fy in LAYOUTS[front]:
        centre = (int(margin + fx * size), int(margin + top_h + fy * (size - top_h)))
        cv2.circle(canvas, centre, size // 16, (235, 235, 235), -1)
    return cv2.GaussianBlur(canvas, (blur | 1, blur | 1), 0)


@pytest.mark.parametrize("value", [1, 2, 3, 4, 5, 6])
def test_reads_every_face(value):
    reading = read_die(synthetic_die(value))
    assert reading.is_die and reading.value == value, reading
    assert describe_die(reading) == f"a die showing {value}"


def test_rejects_things_that_are_not_dice():
    card = np.full((90, 140, 3), (200, 210, 220), dtype=np.uint8)  # a light card: no dark body
    assert not read_die(card).is_die
    standee = np.full((160, 100, 3), (200, 210, 220), dtype=np.uint8)
    cv2.rectangle(standee, (30, 5), (70, 150), (30, 30, 30), -1)  # tall dark thing: not square
    assert not read_die(standee).is_die
    assert not read_die(np.zeros((4, 4, 3), dtype=np.uint8)).is_die
    assert describe_die(read_die(card)) == "not a die"
