"""Thin wrapper around the ``reachy-mini`` SDK with a speaker-only fallback.

    with Robot.connect() as robot:      # real robot through the daemon on REACHY_HOST:REACHY_PORT
        robot.nod()
        robot.say(clip)                 # a Clip from src.speech.tts

    with Robot.connect(simulated=True) as robot:   # no daemon: gestures are logged, audio -> afplay

The SDK is imported lazily so the rest of the project (tests, ingestion) never needs it.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config import REACHY_CONNECTION_MODE, REACHY_HOST, REACHY_MEDIA_BACKEND, REACHY_PORT
from src.logger import get_logger

log = get_logger(__name__)


class Robot:
    """Gestures and speech on a Reachy Mini, or on the Mac speaker when simulated."""

    def __init__(self, mini: Any | None):
        self._mini = mini
        self.simulated = mini is None

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    @contextmanager
    def connect(cls, *, simulated: bool = False, timeout: float = 10.0):
        if simulated:
            log.info("robot simulated: gestures logged, audio through the Mac speaker")
            yield cls(None)
            return
        from reachy_mini import ReachyMini  # imported here so the SDK is optional

        mini = ReachyMini(
            host=REACHY_HOST,
            port=REACHY_PORT,
            connection_mode=REACHY_CONNECTION_MODE,
            media_backend=REACHY_MEDIA_BACKEND,
            timeout=timeout,
        )
        with mini:
            robot = cls(mini)
            robot.wake()
            try:
                yield robot
            finally:
                robot.rest()

    def wake(self) -> None:
        if self.simulated:
            return
        self._mini.enable_motors()
        self._mini.wake_up()
        time.sleep(0.5)

    def rest(self) -> None:
        if self.simulated:
            return
        try:
            self.look(pitch=0, yaw=0, duration=0.8)
        except Exception as exc:  # never fail on the way out
            log.warning("could not return the head to neutral: %s", exc)

    # ------------------------------------------------------------------ gestures
    def look(self, *, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0, duration: float = 0.8) -> None:
        """Move the head to (pitch, yaw, roll) in degrees; positive pitch looks down."""
        if self.simulated:
            log.info("[sim] look pitch=%.0f yaw=%.0f roll=%.0f", pitch, yaw, roll)
            return
        from reachy_mini.utils import create_head_pose

        pose = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
        self._mini.goto_target(head=pose, duration=duration)
        time.sleep(duration)

    def nod(self) -> None:
        """A small "yes": down, up."""
        self.look(pitch=12, duration=0.4)
        self.look(pitch=0, duration=0.4)

    def look_at_table(self) -> None:
        self.look(pitch=35, duration=1.0)

    def antennas(self, left_deg: float, right_deg: float, duration: float = 0.4) -> None:
        if self.simulated:
            log.info("[sim] antennas %.0f/%.0f", left_deg, right_deg)
            return
        import numpy as np

        self._mini.goto_target(antennas=np.deg2rad([left_deg, right_deg]), duration=duration)
        time.sleep(duration)

    # ------------------------------------------------------------------ audio
    def say(self, clip: Any) -> None:
        """Play a WAV clip (``src.speech.tts.Clip``) and block until it finishes."""
        path = Path(clip.path)
        if self.simulated:
            subprocess.run(["afplay", str(path)], check=False)
            return
        self._mini.media.play_sound(str(path))
        time.sleep(float(clip.duration_s) + 0.3)


__all__ = ["Robot"]
