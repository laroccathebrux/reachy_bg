"""Thin wrapper around the ``reachy-mini`` SDK with a speaker-only fallback.

    with Robot.connect() as robot:      # real robot through the daemon on REACHY_HOST:REACHY_PORT
        robot.emotion("curious1")       # a move from Pollen's recorded emotions library
        robot.say(clip)                 # a Clip from src.speech.tts

    with Robot.connect(simulated=True) as robot:   # no daemon: gestures are logged, audio -> afplay

Motion rules learned the hard way:

* Expressive motion comes from the **recorded emotions library** (85 moves animated by
  Pollen, each with an optional sound), never from hand-written pose sequences.
* While the robot speaks, the **daemon's head wobbler** turns the audio into head motion.
  Nothing else sends motion commands during speech; concurrent commands fight the wobbler
  and look mechanical.
* The wobbler is always disabled on the way out (also on Ctrl+C), otherwise the head keeps
  swaying wherever it was left.

The SDK is imported lazily so the rest of the project (tests, ingestion) never needs it.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from src.config import REACHY_CONNECTION_MODE, REACHY_HOST, REACHY_MEDIA_BACKEND, REACHY_PORT
from src.logger import get_logger

log = get_logger(__name__)

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

# A few library moves with a meaning in our conversation states.
THINKING_MOVES = ("thoughtful1", "inquiring1", "curious1")
GREETING_MOVES = ("welcoming1", "welcoming2")
AGREE_MOVES = ("yes1", "understanding1")
OOPS_MOVES = ("oops1", "uncertain1")


class Robot:
    """Gestures and speech on a Reachy Mini, or on the Mac speaker when simulated."""

    def __init__(self, mini: Any | None):
        self._mini = mini
        self.simulated = mini is None
        self._library: Any | None = None
        self._move_thread: threading.Thread | None = None
        self._daemon = f"http://{REACHY_HOST}:{REACHY_PORT}/api"

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    @contextmanager
    def connect(
        cls,
        *,
        simulated: bool = False,
        timeout: float = 10.0,
        media_backend: str = REACHY_MEDIA_BACKEND,
    ):
        if simulated:
            log.info("robot simulated: gestures logged, audio through the Mac speaker")
            yield cls(None)
            return
        from reachy_mini import ReachyMini  # imported here so the SDK is optional

        mini = ReachyMini(
            host=REACHY_HOST,
            port=REACHY_PORT,
            connection_mode=REACHY_CONNECTION_MODE,
            media_backend=media_backend,
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
        """Stop everything that could keep moving and return to a neutral pose."""
        if self.simulated:
            return
        self.stop_speaking()
        self._wait_for_move()
        try:
            self.look(pitch=0, yaw=0, duration=0.8)
            self.antennas(0, 0, duration=0.5)
        except Exception as exc:  # never fail on the way out
            log.warning("could not return to neutral: %s", exc)

    # ------------------------------------------------------------------ basic poses
    def look(
        self,
        *,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        body_yaw: float | None = None,
        duration: float = 0.8,
    ) -> None:
        """Move the head to (pitch, yaw, roll) in degrees; positive pitch looks down.

        ``body_yaw`` (degrees, positive to the left) also turns the body on its base; None
        keeps the current body angle. The head yaw is relative to the body.
        """
        if self.simulated:
            log.info("[sim] look pitch=%.0f yaw=%.0f roll=%.0f body=%s", pitch, yaw, roll, body_yaw)
            return
        from reachy_mini.utils import create_head_pose

        pose = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
        body = None if body_yaw is None else math.radians(body_yaw)
        self._mini.goto_target(head=pose, duration=duration, body_yaw=body)
        time.sleep(duration)

    def look_at_table(self) -> None:
        self.look(pitch=35, duration=1.0)

    def antennas(self, left_deg: float, right_deg: float, duration: float = 0.4) -> None:
        """Move the antennas (degrees); the body stays where it is."""
        if self.simulated:
            log.info("[sim] antennas %.0f/%.0f", left_deg, right_deg)
            return
        import numpy as np

        self._mini.goto_target(antennas=np.deg2rad([left_deg, right_deg]), duration=duration, body_yaw=None)
        time.sleep(duration)

    # ------------------------------------------------------------------ emotions
    def available_emotions(self) -> list[str]:
        return [] if self.simulated else sorted(self._moves().list_moves())

    def emotion(self, name: str, *, sound: bool = True, block: bool = True) -> None:
        """Play a recorded emotion by name (see ``available_emotions``).

        The move's sidecar sound, when present, is played through the daemon (the SDK
        client's own audio path is silent on macOS). ``block=False`` returns immediately and
        the next ``say``/``rest`` waits for the move to finish.
        """
        if self.simulated:
            log.info("[sim] emotion %s", name)
            return
        self._wait_for_move()
        move = self._moves().get(name)
        if sound and move.sound_path is not None:
            self._post("media/play_sound", json={"file": str(move.sound_path)})

        def run() -> None:
            try:
                self._mini.play_move(move, initial_goto_duration=0.4, sound=False)
            except Exception as exc:
                log.warning("emotion %s failed: %s", name, exc)

        self._move_thread = threading.Thread(target=run, name=f"emotion-{name}", daemon=True)
        self._move_thread.start()
        if block:
            self._wait_for_move()

    def _moves(self) -> Any:
        if self._library is None:
            from reachy_mini.motion.recorded_move import RecordedMoves

            self._library = RecordedMoves(EMOTIONS_DATASET)
        return self._library

    def _wait_for_move(self, timeout: float = 15.0) -> None:
        if self._move_thread is not None and self._move_thread.is_alive():
            self._move_thread.join(timeout=timeout)
        self._move_thread = None

    # ------------------------------------------------------------------ speech
    def say(self, clip: Any, *, interrupt: Callable[[], bool] | None = None, poll_s: float = 0.05) -> bool:
        """Play a WAV clip (``src.speech.tts.Clip``) on the robot speaker and block until it finishes.

        Playback goes through the daemon's REST endpoint rather than the SDK client's own
        GStreamer pipeline: on macOS the client-side ``playbin`` reports success but stays
        silent, while the daemon (which owns the audio device) plays reliably. The daemon's
        head wobbler animates the head from the audio for the duration of the clip.

        ``interrupt`` is polled every ``poll_s``; when it returns True the clip is cut short
        (barge-in) and the method returns False. It returns True when the clip played fully.
        """
        path = Path(clip.path).resolve()
        duration = float(clip.duration_s)
        if self.simulated:
            player = subprocess.Popen(["afplay", str(path)])
            try:
                while player.poll() is None:
                    if interrupt is not None and interrupt():
                        player.terminate()
                        return False
                    time.sleep(poll_s)
            finally:
                if player.poll() is None:
                    player.terminate()
            return True
        self._wait_for_move()
        self._post("media/wobbling/enable")
        try:
            if not self._post("media/play_sound", json={"file": str(path)}):
                log.warning("daemon play_sound failed; falling back to the SDK client")
                self._mini.media.play_sound(str(path))
            deadline = time.monotonic() + duration + 0.3
            while time.monotonic() < deadline:
                if interrupt is not None and interrupt():
                    self.stop_speaking()
                    return False
                time.sleep(poll_s)
            return True
        finally:
            self._post("media/wobbling/disable")

    def stop_speaking(self) -> None:
        """Interrupt the current clip and stop the head wobbler."""
        if self.simulated:
            return
        self._post("media/stop_sound")
        self._post("media/wobbling/disable")
        try:
            self._mini.cancel_move()
        except Exception:
            pass

    # ------------------------------------------------------------------ daemon REST
    def _post(self, endpoint: str, *, json: dict | None = None, timeout: float = 10.0) -> bool:
        try:
            httpx.post(f"{self._daemon}/{endpoint}", json=json, timeout=timeout).raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log.warning("daemon %s failed: %s", endpoint, exc)
            return False


__all__ = ["Robot", "THINKING_MOVES", "GREETING_MOVES", "AGREE_MOVES", "OOPS_MOVES"]
