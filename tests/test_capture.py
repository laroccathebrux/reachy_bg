"""Sharpness scoring, sharpest-of-n capture and a sweep on a fake camera."""

import json

import numpy as np

from src.vision.capture import View, capture_sharpest, sharpness, sweep, yaw_views
from src.vision.preview import FakeCamera


def test_sharpness_prefers_detail_over_blur():
    rng = np.random.default_rng(0)
    detailed = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    flat = np.full((60, 80, 3), 120, dtype=np.uint8)
    assert sharpness(detailed) > sharpness(flat) == 0.0
    assert sharpness(np.zeros((2, 2), dtype=np.uint8)) == 0.0


def test_capture_sharpest_picks_the_best_of_distinct_frames():
    rng = np.random.default_rng(1)
    frames = [
        np.full((40, 40, 3), 90, dtype=np.uint8),
        rng.integers(0, 255, (40, 40, 3), dtype=np.uint8),
        np.full((40, 40, 3), 200, dtype=np.uint8),
    ]
    calls = {"n": 0}

    def get_frame():
        i = min(calls["n"] // 2, len(frames) - 1)  # every frame is served twice: the repeat must be skipped
        calls["n"] += 1
        return frames[i]

    best, score = capture_sharpest(get_frame, frames=3, interval_s=0.0, sleep=lambda s: None)
    assert best is frames[1] and score > 0


def test_capture_sharpest_times_out_without_frames():
    clock = {"t": 0.0}

    def tick(_s):
        clock["t"] += 1.0

    best, score = capture_sharpest(lambda: None, timeout_s=2.0, clock=lambda: clock["t"], sleep=tick)
    assert best is None and score == 0.0


def test_yaw_views_names():
    names = [v.name for v in yaw_views([60, 0, -30], pitch=30)]
    assert names == ["left60", "centre", "right30"]
    assert yaw_views([10])[0].pitch == 35.0


def test_sweep_moves_saves_and_indexes(tmp_path):
    camera = FakeCamera(80, 60)
    views = [View("left", body_yaw=40), View("centre"), View("right", body_yaw=-40, pitch=30)]
    seen = []
    captures = sweep(
        camera,
        views,
        out_dir=tmp_path,
        name="sweep_test",
        frames=2,
        sleep=lambda s: None,
        on_view=seen.append,
    )
    assert [c.view.name for c in captures] == ["left", "centre", "right"] and len(seen) == 3
    assert camera.looks == [(35.0, 0.0, 40.0), (35.0, 0.0, 0.0), (30.0, 0.0, -40.0)]
    folder = tmp_path / "sweep_test"
    assert sorted(p.name for p in folder.iterdir()) == ["centre.jpg", "left.jpg", "right.jpg", "views.json"]
    index = json.loads((folder / "views.json").read_text())
    assert index["views"][2]["body_yaw"] == -40.0 and index["views"][2]["width"] == 80
