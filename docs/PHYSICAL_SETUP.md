# Physical Setup and Robot Control

Everything about the table, the robot hardware and how the code moves the head and reads the
camera. Numbers marked *(to measure)* are filled in during Phase 1 calibration.

## Hardware in use

| Item | Detail |
|---|---|
| Robot | **Reachy Mini Lite** (Pollen Robotics / Hugging Face). USB-C data cable to the Mac plus its own 7 V power supply. The Mac runs the daemon. |
| Head | 6-DoF Stewart platform: 3 translations + roll, pitch, yaw. Software limits: pitch and roll +/- 40 deg, head yaw +/- 180 deg, body yaw +/- 160 deg, and at most 65 deg between head and body yaw. |
| Body | 1 yaw axis; 2 antennas (1 axis each). |
| Camera | Raspberry Pi Camera Module 3 Wide (IMX708, 120 deg field of view), exposed as a UVC device. Default capture 1920x1080 at 60 fps; the SDK's local feed delivers about 10 fps. |
| Speaker | 5 W, used for all speech output. |
| Microphone | 4-mic XMOS array on the robot, **not usable: flat cable torn**. A Mac input device is used instead (see [SPEECH_PIPELINE.md](SPEECH_PIPELINE.md)). |
| Host | MacBook Pro M1 Max, 64 GB; Python 3.12 venv managed by `uv`. |
| Serial device | appears as `/dev/cu.usbmodem*` when the robot is plugged in. |

## Connection model

The `reachy-mini` package installs a **daemon** that owns the motors, the camera and the
audio devices and serves a REST + WebSocket API on `http://127.0.0.1:8000` (`/docs` shows it).
The Python `ReachyMini` client talks to that daemon. Starting a session:

```bash
uv run reachy-mini-daemon          # keep it running in its own terminal; takes up to ~1 min
```

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini(media_backend="default") as mini:  # localhost first, then network
    mini.enable_motors()
    mini.wake_up()
    mini.goto_target(head=create_head_pose(pitch=35, degrees=True), duration=1.0)
    frame = mini.media.get_frame()  # BGR ndarray or None while warming up
    mini.media.play_sound("/abs/path/line.wav")  # speaker on the robot
```

Gotchas learned from the SDK docs: after the daemon starts, the robot may be asleep with
motors off and silently ignore commands (`enable_motors()` + `wake_up()` fix it); frames are
`None` for the first seconds; connecting with media enabled and never reading frames logs
buffer-overrun warnings (use `media_backend="no_media"` for motion-only scripts); on macOS the
GStreamer Python plugin can crash with a non-python.org Python build (see the SDK
troubleshooting page). SDK version pinned in `pyproject.toml`.

## Camera preview (placing the robot and the board)

```bash
uv run python -m src.vision.preview          # then open http://127.0.0.1:8090
```

The page streams the camera with guides (thirds, centre cross, a dashed rectangle the board
should fill), moves the head with pitch/yaw sliders and saves full-resolution snapshots to
`data/captures/board/`. Two macOS details. First, start `reachy-mini-daemon` from a terminal
application that has the camera permission (iTerm or Terminal, listed in System Settings >
Privacy & Security > Camera); a daemon started from a Claude Code session is attributed to
the Claude Code helper binary, which cannot ask for the camera, and `data/daemon.log` then
shows `Device video access permission has been denied` and no frames ever arrive (motors and
audio still work, so the problem is easy to miss). Second, nothing else may listen on port
8000 over IPv6, because the SDK dials `localhost` (`::1` first) and another service there
answers instead of the daemon.

## Table layout

```
                         REACHY MINI (on a stable riser)
                                 |
                                 |  camera in the head, pitch ~35 deg down
                                 v
        +----------------------------------------------------------+
        |                                                          |
        |                    GAME BOARD (33 x 22 in)                |
        |   doom/omen tracks on the far edge, reserve near robot    |
        |                                                          |
        +----------------------------------------------------------+
                          players sit on the long sides
```

- Robot placed at the **middle of one short edge**, camera about 45 cm above the table
  *(to measure)*, 20-30 cm from the board edge. With a 120 deg lens the full board should
  fit in one frame from there; verify with the calibration checklist.
- Riser must not wobble: the head moves during speech and face tracking, and the board read
  happens with the head held still.
- The Mac sits next to the robot; its microphone (or a USB microphone) points at the players.
- Lighting: overhead and diffuse. Avoid a window behind the players (backlight) and lamps
  that cast player shadows on the board. Test at the actual play time of day.

## Gaze states

The head has two jobs that conflict: look at people (social) and look at the board (vision).
The integration layer owns a small state machine:

| State | Head | Face tracking | Camera used for |
|---|---|---|---|
| LISTENING | free, tracking the current speaker | on | speaker detection (later) |
| SPEAKING | free, speech-reactive motion | on | - |
| READING_BOARD | held at the table pose (pitch 35 deg, yaw center / +-28 deg for a sweep) | off | board capture |
| THINKING | last pose, still | off | - |

Board reading sequence: switch to READING_BOARD, move with a 1 s minimum-jerk motion, wait
0.5 s for settling, grab 3 frames 50 ms apart, keep the sharpest (Laplacian variance), then
return to LISTENING. Never read the board while speaking: the head moves.

## Calibration checklist (Phase 1, before any vision code)

```
[ ] Riser height and distance recorded: ___ cm / ___ cm
[ ] Head pitch that frames the whole board: ___ deg (start at 35)
[ ] Full board visible in one frame at 1920x1080? yes / no
[ ] Sweep needed (left / center / right)? yes / no
[ ] 10 consecutive captures with the head held: identical framing? yes / no
[ ] Move head, return, capture: framing matches the first? yes / no
[ ] Exposure: tokens readable, no blown highlights (adjust UVC auto-exposure if the image is dark)
[ ] Lighting recorded (time of day, lamps on/off)
[ ] Reference photos of every base-game token type saved under data/captures/reference/
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Robot does not move, no error | motors disabled / asleep | `enable_motors()`, `wake_up()` |
| Daemon never opens port 8000 | serial port busy or wrong | unplug/replug, check `/dev/cu.usbmodem*`, kill old daemons |
| Frames always `None` | media not started or another process holds the camera | wait 20 s; ensure only one daemon; `media_backend="default"` |
| Image dark | UVC auto-exposure off | set auto-exposure priority via the SDK camera controller |
| Blurry board reads | head not settled or riser wobble | longer settle, stiffer riser, sharpest-of-3 selection |
| Robot cannot be interrupted while speaking | echo cancellation absent | use the Mac input device with system echo cancellation, not the robot mic |
