"""Watching one view of the board and reporting what moved, without naming anything.

    uv run python -m src.vision.motion                  # the robot watches the centre view
    uv run python -m src.vision.motion --from-preview   # read a running preview's stream instead
    uv run python -m src.vision.motion --fake           # synthetic frames, no robot

    watcher = MotionWatcher(registration)
    move = watcher.feed(frame)                          # None until the board settles again
    print(move.describe())                              # "a piece left Buenos Aires, one arrived at Rome"

The scan in ``detect.py`` answers "what is on the board" against a baseline of the *empty*
board, which is captured once and ages badly: the evening baseline was useless the next
morning (board brightness 138 -> 86, median difference 28 against 2 the night before) and the
scan invented pieces. This module answers a narrower question that does not age: "what changed
just now". Consecutive frames are 0.1 s apart, so the light between them is the same light.

Measured on the robot's live stream at 960x540, board untouched (2026-09-16):

| Frames apart | Median difference | 95th percentile | Max |
|---|---|---|---|
| 1 (0.10 s) | 2.24 | 4.58 | 14.7 |
| 9 (0.94 s) | 2.24 | 4.58 | 14.9 |

A piece is worth 33-60 by the same measure, so the signal sits about 7x above the noise, and
the noise does not grow with the interval (it is the sensor, not drifting light).

The *activity* that triggers a reading was measured the same way, on the same stream: an
untouched board scores exactly 0.0000 (no pixel differs by the 30 grey levels ``activity``
counts), and a hand moving pieces peaks at 0.006 to 0.039 of the frame. The gap is the whole
margin, and it is one-sided: the floor is zero, so the trigger sits just above it.

How a move is read: while the board is quiet the *anchor* frame is kept fresh. A hand reaching
in raises the activity far above anything a piece does, which is the cue that a move is
happening; when the board is quiet again for ``settle_frames``, the anchor is compared with the
new frame by ``find_pieces``. The occlusion is therefore not a problem to work around but the
trigger itself.

Which blob left and which arrived does *not* follow from that difference: it is symmetric, so
using either frame as the other's baseline returns the same blobs. The tie is broken by
``stands_out``, which asks how far each blob's colour sits from the board immediately around
it, in each of the two frames: a piece stands out against the board, a vacated space does not.

The view is fixed: the registration is computed once, with the robot still. Moving the head or
the body invalidates it, so the caller pauses the watcher across a sweep and re-registers after.

Only one process can hold the camera, and during a session that process is the preview. So the
frames can also come from the preview's own MJPEG stream (``--from-preview``): the page stays
up and visible while a piece is moved, at the cost of the stream's 960-wide frames, which is the
resolution the noise above was measured at.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.logger import get_logger
from src.vision.detect import MAP_WIDTH, Baseline, Piece, difference_map, find_pieces

log = get_logger(__name__)

ACTIVITY_SCALE = 480  # frames are compared at this width: 4x less work, same activity fraction
ACTIVITY_LEVEL = 30  # per-pixel grey difference counted as movement (sensor noise peaks at 15)
# Measured on the live 960x540 stream (2026-09-16), not estimated: with ACTIVITY_LEVEL at 30 no
# pixel of an untouched board passes at all, so a still board reads as exactly 0.0000 for
# minutes at a time, while a hand moving pieces peaks between 0.006 and 0.039 per frame. The
# first guess at these bounds was 0.04 / 0.004, and a real move topped out at 0.0386: the
# trigger never fired. The floor is what matters, and the floor is zero.
DISTURBED_FRACTION = 0.005  # of the frame moving: something is happening over the board
QUIET_FRACTION = 0.001  # below this the board is considered still
SETTLE_FRAMES = 5  # consecutive quiet frames before a verdict (~0.5 s at 9.5 fps)
MIN_MOVE_AREA = 300  # rectified-map pixels, as in detect.MIN_AREA
# A sweep of the body scored 0.036 to 0.100 per frame and a hand 0.006 to 0.039: the two ranges
# overlap, so how much moved cannot say whether it was the camera. Where the board sits in the
# frame can, and that is what VIEW_SHIFT_PX checks before a verdict is trusted.
VIEW_SHIFT_PX = 25.0  # board corners may wander this far in the frame before the view is stale
MAX_PIECES_PER_MOVE = 4  # a verdict naming more than this is not a person moving pieces


@dataclass
class Move:
    """What changed between two quiet moments: pieces that left, pieces that arrived."""

    departed: list[Piece] = field(default_factory=list)
    arrived: list[Piece] = field(default_factory=list)
    seconds: float = 0.0  # how long the board was disturbed
    at: float = field(default_factory=time.time)

    @property
    def empty(self) -> bool:
        return not self.departed and not self.arrived

    @staticmethod
    def _where(piece: Piece) -> str:
        return piece.space or (f"near {piece.near}" if piece.near else "off the spaces")

    def describe(self) -> str:
        """One line in plain English, naming spaces only (this module never names pieces)."""
        if self.empty:
            return "nothing changed"
        left = [self._where(p) for p in self.departed]
        came = [self._where(p) for p in self.arrived]
        if len(left) == 1 and len(came) == 1:
            return f"a piece moved from {left[0]} to {came[0]}"
        parts = []
        if left:
            parts.append(f"{len(left)} piece(s) left {', '.join(left)}")
        if came:
            parts.append(f"{len(came)} piece(s) arrived at {', '.join(came)}")
        return "; ".join(parts)

    def record(self) -> dict[str, Any]:
        return {
            "at": round(self.at, 3),
            "seconds": round(self.seconds, 2),
            "departed": [p.record() for p in self.departed],
            "arrived": [p.record() for p in self.arrived],
            "text": self.describe(),
        }


def activity(previous: np.ndarray, current: np.ndarray, *, level: int = ACTIVITY_LEVEL) -> float:
    """Fraction of the frame whose grey value moved by more than ``level`` between two frames."""
    import cv2

    if previous.shape != current.shape:
        return 1.0
    scale = ACTIVITY_SCALE / max(1, current.shape[1])
    size = (max(1, int(current.shape[1] * scale)), max(1, int(current.shape[0] * scale)))
    a = cv2.cvtColor(cv2.resize(previous, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(current, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(a, b) > level))


RING = 1.8  # the local background is read from a box this many times the blob's own box


def stands_out(rectified: np.ndarray, piece: Piece, *, ring: float = RING) -> float:
    """How far the blob's colour sits from the board immediately around it, in Lab.

    The difference between two frames is symmetric: the same two blobs show up whichever frame
    is used as the baseline, so it cannot say by itself which one holds the piece. This can: a
    piece stands out against the board around it, an empty space does not.
    """
    import cv2

    h, w = rectified.shape[:2]
    bw, bh = max(4.0, piece.width), max(4.0, piece.height)

    def box(factor: float) -> tuple[int, int, int, int]:
        return (
            int(max(0, piece.x - bw * factor / 2)),
            int(max(0, piece.y - bh * factor / 2)),
            int(min(w, piece.x + bw * factor / 2)),
            int(min(h, piece.y + bh * factor / 2)),
        )

    x0, y0, x1, y1 = box(1.0)
    rx0, ry0, rx1, ry1 = box(ring)
    if x1 <= x0 or y1 <= y0 or rx1 <= rx0 or ry1 <= ry0:
        return 0.0
    inner = cv2.cvtColor(rectified[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    around = cv2.cvtColor(rectified[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2LAB).astype(np.float32)
    # The ring is the larger share of the wider box, so its median is the local board colour.
    background = np.median(around.reshape(-1, 3), axis=0)
    return float(np.linalg.norm(inner.reshape(-1, 3).mean(axis=0) - background))


def occupancy(rectified: np.ndarray, baseline: Baseline, piece: Piece) -> float:
    """Mean difference from the *empty* board inside the blob's box: high means a piece is there.

    This is the reliable arbiter of direction, and it is absolute rather than relative: it asks
    "does this spot differ from the bare board" instead of "does this spot differ from its
    surroundings". ``stands_out`` gets that wrong wherever the board's own art is busy - a piece
    landing on the dark "The Heart of Africa" card barely raises the local contrast, and a real
    move from The Pyramids to it was reported as two departures.

    The baseline only arbitrates; it never triggers. A stale one degrades this verdict but
    cannot make the watcher fire, so the module keeps working in light the baseline never saw.
    """
    h, w = rectified.shape[:2]
    x0 = int(max(0, piece.x - piece.width / 2))
    x1 = int(min(w, piece.x + piece.width / 2))
    y0 = int(max(0, piece.y - piece.height / 2))
    y1 = int(min(h, piece.y + piece.height / 2))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    diff = difference_map(rectified, baseline)
    covered = baseline.coverage[y0:y1, x0:x1] == 255
    window = diff[y0:y1, x0:x1]
    return float(window[covered].mean()) if covered.any() else 0.0


def compare(
    registration: Any,
    before: np.ndarray,
    after: np.ndarray,
    *,
    min_area: int = MIN_MOVE_AREA,
    baseline: Baseline | None = None,
) -> Move:
    """What left and what arrived between two frames of the same fixed view.

    Each frame is used as the other's baseline, so the whole of ``find_pieces`` applies: the
    adaptive threshold, the hysteresis that keeps a standee's footprint, the masked Reserve and
    legend, and the naming of a piece by the space under its base. Both directions return the
    same blobs, so each blob is then assigned to the frame it actually holds the piece in, which
    also keeps the crop taken from the frame where the piece is visible.

    With a ``baseline`` of the empty board the assignment is made by ``occupancy`` (absolute);
    without one it falls back to ``stands_out`` (relative, and wrong over busy board art).
    """
    seen_after = find_pieces(registration, after, Baseline.capture(registration, before), min_area=min_area)
    seen_before = find_pieces(registration, before, Baseline.capture(registration, after), min_area=min_area)
    width = baseline.image.shape[1] if baseline is not None else MAP_WIDTH
    rect_before = registration.rectify(before, width=width)
    rect_after = registration.rectify(after, width=width)
    if baseline is not None:

        def holds_piece(rectified: np.ndarray, piece: Piece) -> float:
            return occupancy(rectified, baseline, piece)
    else:

        def holds_piece(rectified: np.ndarray, piece: Piece) -> float:
            return stands_out(rectified, piece)

    arrived = [p for p in seen_after if holds_piece(rect_after, p) >= holds_piece(rect_before, p)]
    departed = [p for p in seen_before if holds_piece(rect_before, p) > holds_piece(rect_after, p)]
    return Move(departed=departed, arrived=arrived)


class MotionWatcher:
    """Feeds frames of one fixed view; returns a ``Move`` each time the board settles after a change."""

    def __init__(
        self,
        registration: Any,
        *,
        settle_frames: int = SETTLE_FRAMES,
        disturbed_fraction: float = DISTURBED_FRACTION,
        quiet_fraction: float = QUIET_FRACTION,
        min_area: int = MIN_MOVE_AREA,
        baseline: Baseline | None = None,
        reference: Any = None,
        view_shift_px: float = VIEW_SHIFT_PX,
        max_pieces: int = MAX_PIECES_PER_MOVE,
    ):
        self.registration = registration
        self.settle_frames = settle_frames
        self.disturbed_fraction = disturbed_fraction
        self.quiet_fraction = quiet_fraction
        self.min_area = min_area
        self.baseline = baseline
        self.reference = reference  # BoardReference, to re-register and notice the view moving
        self.view_shift_px = view_shift_px
        self.max_pieces = max_pieces
        self.anchor: np.ndarray | None = None  # last frame of the board at rest
        self.previous: np.ndarray | None = None
        self.disturbed = False
        self.quiet_run = 0
        self.last_activity = 0.0
        self.disturbed_since = 0.0
        self.moves: list[Move] = []

    @property
    def state(self) -> str:
        return "disturbed" if self.disturbed else "quiet"

    def reset(self, frame: np.ndarray | None = None) -> None:
        """Forget the history; call after the robot moved and the view was registered again."""
        self.anchor = None if frame is None else frame.copy()
        self.previous = self.anchor
        self.disturbed = False
        self.quiet_run = 0
        self.camera_moved = False

    def view_shifted(self, frame: np.ndarray) -> bool:
        """Has the board moved within the frame since the registration was taken?

        How *much* of the picture changed cannot answer this: a hand reaches 0.039 and a body
        sweep starts at 0.036. Where the board's corners sit can, so the frame is registered
        again and the outlines compared. Without a ``reference`` to register with, the watcher
        trusts its view and leans on ``max_pieces`` to catch the damage instead.
        """
        if self.reference is None:
            return False
        fresh = self.reference.locate(frame)
        if fresh is None:
            log.warning("motion: the board is no longer recognised in this view")
            return True
        shift = float(np.abs(self.registration.outline() - fresh.outline()).max())
        if shift <= self.view_shift_px:
            return False
        log.warning(
            "motion: the view moved (board corners by %.0f px); the registration is stale, "
            "not reading a move until it is renewed",
            shift,
        )
        return True

    def feed(self, frame: np.ndarray) -> Move | None:
        """One frame in; a ``Move`` out when the board has just settled after being disturbed."""
        if self.previous is None:
            self.anchor = frame.copy()
            self.previous = frame.copy()
            return None
        self.last_activity = activity(self.previous, frame)
        self.previous = frame.copy()
        if self.last_activity >= self.disturbed_fraction:
            if not self.disturbed:
                self.disturbed_since = time.time()
                log.info("motion: board disturbed (activity %.3f)", self.last_activity)
            self.disturbed = True
            self.quiet_run = 0
            return None
        if self.last_activity > self.quiet_fraction:
            self.quiet_run = 0  # neither clearly moving nor clearly still: wait
            return None
        self.quiet_run += 1
        if not self.disturbed:
            self.anchor = frame.copy()  # nothing happened; keep the anchor fresh
            return None
        if self.quiet_run < self.settle_frames:
            return None
        if self.view_shifted(frame):
            # Settled, but the board is no longer where the registration says it is: the robot
            # turned. Every verdict taken here would be noise (a sweep once produced "12 pieces
            # left"), so drop it, re-anchor, and let the caller register the new view.
            self.anchor = frame.copy()
            self.disturbed = False
            self.quiet_run = 0
            return None
        move = compare(self.registration, self.anchor, frame, min_area=self.min_area, baseline=self.baseline)
        move.seconds = time.time() - self.disturbed_since
        self.anchor = frame.copy()
        self.disturbed = False
        self.quiet_run = 0
        total = len(move.departed) + len(move.arrived)
        if total > self.max_pieces:
            # A human moves one or two pieces at a time. A verdict naming a dozen means the
            # picture changed for some other reason (the view shifted, the light jumped), and
            # reporting it as a move would be worse than saying nothing.
            log.warning("motion: %d pieces changed at once; ignoring as not a move", total)
            return None
        if move.empty:
            log.info("motion: board settled, nothing changed (%.1f s)", move.seconds)
            return None
        log.info("motion: %s (%.1f s)", move.describe(), move.seconds)
        self.moves.append(move)
        return move


JPEG_START, JPEG_END = b"\xff\xd8", b"\xff\xd9"


def iter_jpegs(stream: Any, *, chunk: int = 16384) -> Any:
    """Yield the JPEG payloads of a multipart MJPEG byte stream, one per frame.

    The parts are found by the JPEG markers rather than by the multipart boundary: the boundary
    is a detail of the server, the markers are in the data itself.
    """
    buf = b""
    while True:
        data = stream.read(chunk)
        if not data:
            return
        buf += data
        while True:
            start = buf.find(JPEG_START)
            end = buf.find(JPEG_END, start + 2) if start >= 0 else -1
            if start < 0 or end < 0:
                break
            yield buf[start : end + 2]
            buf = buf[end + 2 :]


class PreviewStream:
    """Frames from a running preview's MJPEG stream, so the camera is not opened twice.

    The preview already holds the camera, and a second client of the daemon would fight it for
    the device. Reading its stream instead costs resolution (the stream is downscaled to 960
    wide) but that is where the noise was measured, and it lets the owner watch the page while
    moving a piece. A reader thread keeps only the newest frame, so a slow consumer never
    works its way through a backlog of stale ones.
    """

    def __init__(self, url: str = "http://127.0.0.1:8090/stream", *, timeout: float = 30.0):
        import threading

        self.url = url
        self.timeout = timeout
        self._frame: np.ndarray | None = None
        self._stop = False
        self.frames = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import urllib.error
        import urllib.request

        import cv2

        while not self._stop:
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as stream:
                    for payload in iter_jpegs(stream):
                        if self._stop:
                            return
                        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                        if image is not None:
                            self._frame = image
                            self.frames += 1
                            self.error = None
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self.error = str(exc)
                if self._stop:
                    return
                log.warning("preview stream %s: %s; retrying", self.url, exc)
                time.sleep(1.0)

    def wait_for_frame(self, timeout: float = 15.0) -> np.ndarray | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._frame is not None:
                return self._frame
            time.sleep(0.1)
        return None

    def get_frame(self) -> np.ndarray | None:
        return self._frame

    def look(self, *args: Any, **kwargs: Any) -> None:
        """The preview owns the robot; this source only watches."""

    def close(self) -> None:
        self._stop = True


def watch(
    camera: Any,
    registration: Any,
    *,
    limit: int = 0,
    period_s: float = 0.1,
    show_activity: bool = False,
    watcher: MotionWatcher | None = None,
) -> list[Move]:
    """Print every move until ``limit`` of them (0 = forever); returns what was seen.

    ``show_activity`` prints the peak activity once a second. The thresholds depend on the room
    and the light, so this is how they are checked before trusting a session: an untouched board
    should read 0.0000, and a hand over it should clear ``DISTURBED_FRACTION``.
    """
    watcher = watcher or MotionWatcher(registration)
    bucket_start, peak = time.time(), 0.0
    seen: list[Move] = []
    last_count = -1
    while True:
        # A source that keeps only the newest frame hands out the same one between captures;
        # feeding it twice would read as a still board and settle the watcher early.
        count = getattr(camera, "frames", None)
        frame = camera.get_frame()
        if frame is None or (count is not None and count == last_count):
            time.sleep(period_s)
            continue
        last_count = count
        move = watcher.feed(frame)
        if show_activity:
            peak = max(peak, watcher.last_activity)
            if time.time() - bucket_start >= 1.0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] activity peak {peak:.4f}  {watcher.state}",
                    flush=True,
                )
                bucket_start, peak = time.time(), 0.0
        if move is not None:
            seen.append(move)
            print(f"[{time.strftime('%H:%M:%S')}] {move.describe()}", flush=True)
            if limit and len(seen) >= limit:
                return seen
        time.sleep(period_s)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.vision.board_map import BoardReference
    from src.vision.detect import BaselineSet
    from src.vision.preview import FakeCamera, RobotCamera

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="synthetic frames, no robot")
    parser.add_argument(
        "--from-preview",
        nargs="?",
        const="http://127.0.0.1:8090/stream",
        default=None,
        metavar="URL",
        help="read a running preview's MJPEG stream instead of opening the camera "
        "(default http://127.0.0.1:8090/stream); the preview keeps the robot and stays visible",
    )
    parser.add_argument("--pitch", type=float, default=35.0, help="head pitch, degrees below level")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many moves (0 = forever)")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="ignore the empty-board baseline and decide direction by local contrast instead",
    )
    parser.add_argument(
        "--show-activity",
        action="store_true",
        help="print the peak activity once a second, to check the thresholds against this room",
    )
    parser.add_argument(
        "--disturbed",
        type=float,
        default=DISTURBED_FRACTION,
        help=f"fraction of the frame moving that counts as a disturbance (default {DISTURBED_FRACTION})",
    )
    args = parser.parse_args(argv)

    direct = not args.fake and not args.from_preview
    camera: Any
    if args.from_preview:
        camera = PreviewStream(args.from_preview)
    elif args.fake:
        camera = FakeCamera()
    else:
        camera = RobotCamera()
    try:
        if direct:
            camera.look(args.pitch, 0.0, 0.0)
            time.sleep(1.0)
        frame = None
        for _ in range(75):
            frame = camera.get_frame()
            if frame is not None:
                break
            time.sleep(0.2)
        if frame is None:
            where = args.from_preview or "the camera"
            print(f"no frame from {where}" + (f" ({camera.error})" if getattr(camera, "error", None) else ""))
            return 1
        reference = BoardReference()
        registration = reference.locate(frame)
        if registration is None:
            print("the board was not recognised in this view", flush=True)
            return 1
        baseline = None
        if not args.no_baseline:
            from src.config import CAPTURE_DIR

            try:
                baseline = BaselineSet.load(CAPTURE_DIR / "board_baseline").views.get("centre")
            except (FileNotFoundError, OSError) as exc:
                log.info("no empty-board baseline (%s); direction falls back to local contrast", exc)
        print(
            "direction arbiter: " + ("empty-board baseline" if baseline is not None else "local contrast"),
            flush=True,
        )
        print(
            f"watching: {registration.inliers} inliers, {frame.shape[1]}x{frame.shape[0]}. "
            "Move a piece; Ctrl-C to stop.",
            flush=True,
        )
        watch(
            camera,
            registration,
            limit=args.limit,
            show_activity=args.show_activity,
            watcher=MotionWatcher(
                registration,
                disturbed_fraction=args.disturbed,
                baseline=baseline,
                reference=reference,
            ),
        )
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        camera.close()
    return 0


__all__ = [
    "ACTIVITY_LEVEL",
    "Move",
    "MotionWatcher",
    "PreviewStream",
    "activity",
    "compare",
    "iter_jpegs",
    "occupancy",
    "stands_out",
    "watch",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
