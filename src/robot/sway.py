"""Speech-synced head sway driven by the loudness of the audio being played.

The daemon's own wobbler only reacts to audio played through the daemon; agent audio goes
straight to the USB speaker, so this thread moves the head instead: a few slow sines on
pitch, yaw and roll, scaled by a smoothed loudness envelope, sent as short ``goto_target``
moves at 10 Hz. Nothing else sends head motion while it runs (emotions pause it).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any

from src.logger import get_logger

log = get_logger(__name__)

QUIET_DB = -45.0
LOUD_DB = -20.0
RATE_HZ = 10.0


def envelope(level_db: float, previous: float, *, attack: float = 0.5, release: float = 0.12) -> float:
    """Loudness in [0, 1] with a fast attack and a slow release (pure; unit-tested)."""
    target = min(1.0, max(0.0, (level_db - QUIET_DB) / (LOUD_DB - QUIET_DB)))
    alpha = attack if target > previous else release
    return previous + alpha * (target - previous)


def pose_at(t: float, amount: float) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees for time ``t`` and envelope ``amount``."""
    pitch = 4.0 * amount * math.sin(2 * math.pi * 2.0 * t)
    yaw = 6.0 * amount * math.sin(2 * math.pi * 0.55 * t)
    roll = 2.5 * amount * math.sin(2 * math.pi * 1.3 * t + 1.0)
    return pitch, yaw, roll


class HeadSway:
    def __init__(self, mini: Any | None, level_source: Callable[[], float]):
        self._mini = mini
        self._level = level_source
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self.amount = 0.0

    def start(self) -> HeadSway:
        if self._mini is None:
            return self
        self._thread = threading.Thread(target=self._run, name="head-sway", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def _run(self) -> None:

        period = 1.0 / RATE_HZ
        t0 = time.monotonic()
        idle_sent = True
        while not self._stop.is_set():
            time.sleep(period)
            if self._pause.is_set():
                continue
            self.amount = envelope(self._level(), self.amount)
            if self.amount < 0.02:
                if not idle_sent:
                    self._goto(0.0, 0.0, 0.0, 0.6)
                    idle_sent = True
                continue
            idle_sent = False
            pitch, yaw, roll = pose_at(time.monotonic() - t0, self.amount)
            self._goto(pitch, yaw, roll, period * 1.5)

    def _goto(self, pitch: float, yaw: float, roll: float, duration: float) -> None:
        from reachy_mini.utils import create_head_pose

        try:
            pose = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
            self._mini.goto_target(head=pose, duration=duration, body_yaw=None)
        except Exception as exc:
            log.debug("sway goto failed: %s", exc)


__all__ = ["HeadSway", "envelope", "pose_at"]
