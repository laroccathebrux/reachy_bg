"""Thin wrapper around the ``reachy-mini`` SDK with a speaker-only fallback.

    with Robot.connect() as robot:      # real robot through the daemon on REACHY_HOST:REACHY_PORT
        robot.nod()
        robot.say(clip)                 # a Clip from src.speech.tts

    with Robot.connect(simulated=True) as robot:   # no daemon: gestures are logged, audio -> afplay

The SDK is imported lazily so the rest of the project (tests, ingestion) never needs it.
"""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

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

    def antennas(
        self, left_deg: float, right_deg: float, duration: float = 0.4, *, wait: bool = True
    ) -> None:
        """Move the antennas (degrees). ``body_yaw=None`` keeps the body where it is."""
        if self.simulated:
            log.info("[sim] antennas %.0f/%.0f", left_deg, right_deg)
            return
        import numpy as np

        self._mini.goto_target(antennas=np.deg2rad([left_deg, right_deg]), duration=duration, body_yaw=None)
        if wait:
            time.sleep(duration)

    def thinking(self) -> None:
        """Antennas up and slightly apart: "give me a second"."""
        self.antennas(35, -35, duration=0.5)

    def neutral(self) -> None:
        self.antennas(0, 0, duration=0.5)

    def _wiggle_antennas(self, seconds: float, stop: threading.Event) -> None:
        """Alternate the antennas while the robot talks (runs in a background thread)."""
        import numpy as np

        poses = [(25.0, -10.0), (10.0, -25.0), (30.0, -30.0), (15.0, -15.0)]
        deadline = time.monotonic() + seconds
        i = 0
        while time.monotonic() < deadline and not stop.is_set():
            left, right = poses[i % len(poses)]
            try:
                self._mini.goto_target(antennas=np.deg2rad([left, right]), duration=0.45, body_yaw=None)
            except Exception as exc:  # a failed wiggle must never interrupt speech
                log.debug("antenna wiggle skipped: %s", exc)
                return
            stop.wait(0.55)
            i += 1

    # ------------------------------------------------------------------ audio
    def say(self, clip: Any) -> None:
        """Play a WAV clip (``src.speech.tts.Clip``) on the robot speaker and block until it finishes.

        Playback goes through the daemon's REST endpoint rather than the SDK client's own
        GStreamer pipeline: on macOS the client-side ``playbin`` reports success but stays
        silent, while the daemon (which owns the audio device) plays reliably.
        """
        path = Path(clip.path).resolve()
        if self.simulated:
            subprocess.run(["afplay", str(path)], check=False)
            return
        base = f"http://{REACHY_HOST}:{REACHY_PORT}/api/media"
        duration = float(clip.duration_s)
        # The daemon's head wobbler makes the head move with the audio it plays, and a
        # background thread keeps the antennas alive: the robot visibly "talks".
        self._post(f"{base}/wobbling/enable")
        stop = threading.Event()
        wiggler = threading.Thread(target=self._wiggle_antennas, args=(duration, stop), daemon=True)
        wiggler.start()
        try:
            response = httpx.post(f"{base}/play_sound", json={"file": str(path)}, timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("daemon play_sound failed (%s); falling back to the SDK client", exc)
            self._mini.media.play_sound(str(path))
        time.sleep(duration + 0.3)
        stop.set()
        wiggler.join(timeout=1.0)
        self._post(f"{base}/wobbling/disable")
        self.neutral()

    @staticmethod
    def _post(url: str) -> None:
        try:
            httpx.post(url, timeout=5.0).raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("%s failed: %s", url.rsplit("/api/", 1)[-1], exc)

    def stop_speaking(self) -> None:
        """Interrupt the current clip."""
        if self.simulated:
            return
        try:
            httpx.post(f"http://{REACHY_HOST}:{REACHY_PORT}/api/media/stop_sound", timeout=5.0)
        except httpx.HTTPError as exc:
            log.warning("daemon stop_sound failed: %s", exc)


__all__ = ["Robot"]
