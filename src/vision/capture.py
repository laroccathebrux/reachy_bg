"""Sharp captures of the table, one view at a time, and sweeps across the board.

    frame, score = capture_sharpest(camera.get_frame, frames=3)      # least motion blur of three
    results = sweep(camera, [View("left", body_yaw=40), View("centre"), View("right", body_yaw=-40)])

The board is larger than one camera view at a resolution that still shows card text, so the
robot looks at it in pieces: the body turns on its base (``body_yaw``, positive to the left)
and the head tilts (``pitch``, positive down); every :class:`View` is one such pose. A sweep
moves to each view, waits for the head to settle, keeps the sharpest of a few frames and
saves it under ``data/captures/board/<sweep>/<view>.jpg`` together with a ``views.json``
index (pose, sharpness, size) for the detector and the board map later.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import CAPTURE_DIR
from src.logger import get_logger

log = get_logger(__name__)

BOARD_CAPTURE_DIR = CAPTURE_DIR / "board"
SETTLE_S = 0.6  # after the move ends: let the head stop swinging and the auto-exposure adapt


@dataclass(frozen=True)
class View:
    name: str
    body_yaw: float = 0.0  # degrees, positive turns left
    pitch: float = 35.0  # degrees, positive looks down
    yaw: float = 0.0  # head yaw relative to the body


@dataclass
class Capture:
    view: View
    path: Path
    sharpness: float
    width: int
    height: int

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self.view),
            "path": str(self.path),
            "sharpness": round(self.sharpness, 1),
            "width": self.width,
            "height": self.height,
        }


def sharpness(frame: np.ndarray) -> float:
    """Variance of a Laplacian on the grey image: motion blur and defocus lower it."""
    if frame.ndim == 3:
        grey = frame[:, :, :3].astype(np.float32).mean(axis=2)
    else:
        grey = frame.astype(np.float32)
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    lap = -4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
    return float(lap.var())


def capture_sharpest(
    get_frame: Callable[[], np.ndarray | None],
    *,
    frames: int = 3,
    interval_s: float = 0.15,
    timeout_s: float = 3.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray | None, float]:
    """The sharpest of ``frames`` distinct frames taken ``interval_s`` apart; (None, 0) on timeout."""
    best, best_score, last, taken = None, -1.0, None, 0
    deadline = clock() + timeout_s
    while taken < frames and clock() < deadline:
        frame = get_frame()
        if frame is None or frame is last:
            sleep(0.02)
            continue
        last = frame
        taken += 1
        score = sharpness(frame)
        if score > best_score:
            best, best_score = frame, score
        if taken < frames:
            sleep(interval_s)
    return best, max(best_score, 0.0)


def sweep(
    camera: Any,
    views: list[View],
    *,
    out_dir: Path | None = None,
    name: str | None = None,
    frames: int = 3,
    settle_s: float = SETTLE_S,
    sleep: Callable[[float], None] = time.sleep,
    on_view: Callable[[Capture], None] | None = None,
) -> list[Capture]:
    """Visit every view, save the sharpest frame of each, write ``views.json``; returns the captures.

    ``camera`` needs ``look(pitch, yaw, body_yaw)`` and ``get_frame()`` (the preview's cameras do).
    """
    from src.vision.preview import encode_jpeg

    name = name or time.strftime("sweep_%Y%m%d_%H%M%S")
    folder = (out_dir or BOARD_CAPTURE_DIR) / name
    folder.mkdir(parents=True, exist_ok=True)
    captures: list[Capture] = []
    for view in views:
        camera.look(view.pitch, view.yaw, view.body_yaw)
        sleep(settle_s)
        frame, score = capture_sharpest(camera.get_frame, frames=frames, sleep=sleep)
        if frame is None:
            log.warning("view %s: no frame", view.name)
            continue
        path = folder / f"{view.name}.jpg"
        path.write_bytes(encode_jpeg(frame, quality=92))
        capture = Capture(view, path, score, int(frame.shape[1]), int(frame.shape[0]))
        captures.append(capture)
        log.info(
            "view %s (body %.0f, pitch %.0f): sharpness %.0f -> %s",
            view.name,
            view.body_yaw,
            view.pitch,
            score,
            path,
        )
        if on_view is not None:
            on_view(capture)
    (folder / "views.json").write_text(
        json.dumps({"name": name, "views": [c.record() for c in captures]}, indent=2), encoding="utf-8"
    )
    return captures


def yaw_views(body_yaws: list[float], pitch: float = 35.0) -> list[View]:
    """One view per body angle, named by the angle ("left40", "centre", "right40")."""
    views = []
    for angle in body_yaws:
        if abs(angle) < 0.5:
            label = "centre"
        else:
            label = f"{'left' if angle > 0 else 'right'}{abs(int(round(angle)))}"
        views.append(View(label, body_yaw=float(angle), pitch=float(pitch)))
    return views


__all__ = ["View", "Capture", "sharpness", "capture_sharpest", "sweep", "yaw_views", "BOARD_CAPTURE_DIR"]
