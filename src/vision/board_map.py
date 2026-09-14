"""Where is the board in this frame? Register a camera frame to the canonical map by its own art.

    board = BoardReference()                     # BOARD_REFERENCE_IMAGE, SIFT features computed once
    reg = board.locate(frame)                    # None when the board is not recognised
    reg.space_at(x, y)                           # -> (Space, distance) for a pixel of the frame
    reg.space_pixels()                           # -> {"London": (x, y), ...} for the spaces in view
    reg.rectify(frame, width=1200)               # the frame warped to the top-down map

No markers: the printed map itself is the marker. SIFT keypoints of the reference picture are
matched (Lowe ratio test) against the frame and a homography is fitted by RANSAC; a
registration is accepted when it has ``min_inliers`` consistent matches. Measured on the
robot's 1920x1080 frames against a 1200x790 board picture: about 100 inliers for the centre
view, 40 for the views turned 45 degrees, 0.2 s per frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import BOARD_REFERENCE_IMAGE
from src.logger import get_logger
from src.vision.spaces import SPACES, Space, nearest_space

log = get_logger(__name__)


@dataclass
class Registration:
    """A homography from frame pixels to reference pixels, with what it was built from."""

    homography: np.ndarray  # 3x3, frame -> reference pixels
    inliers: int
    matches: int
    seconds: float
    reference_size: tuple[int, int]  # width, height
    frame_size: tuple[int, int]

    def to_reference(self, points: Any) -> np.ndarray:
        """Frame pixels -> reference pixels (N x 2)."""
        import cv2

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def to_frame(self, points: Any) -> np.ndarray:
        """Reference pixels -> frame pixels (N x 2)."""
        import cv2

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.homography)).reshape(-1, 2)

    def to_map(self, points: Any) -> np.ndarray:
        """Frame pixels -> normalised map coordinates (0..1 of the board width and height)."""
        w, h = self.reference_size
        return self.to_reference(points) / np.array([w, h], dtype=np.float32)

    def space_at(self, x: float, y: float) -> tuple[Space | None, float]:
        """The space under a frame pixel (None when it lies between spaces)."""
        mx, my = self.to_map([(x, y)])[0]
        return nearest_space(float(mx), float(my))

    def outline(self) -> np.ndarray:
        """The board rectangle as a polygon in frame pixels (4 x 2)."""
        w, h = self.reference_size
        return self.to_frame([(0, 0), (w, 0), (w, h), (0, h)])

    def space_pixels(self, *, margin: int = 0) -> dict[str, tuple[float, float]]:
        """Frame pixels of every space centre that falls inside the frame."""
        w, h = self.reference_size
        fw, fh = self.frame_size
        centres = self.to_frame([s.pixel(w, h) for s in SPACES])
        result = {}
        for space, (x, y) in zip(SPACES, centres, strict=True):
            if -margin <= x < fw + margin and -margin <= y < fh + margin:
                result[space.name] = (float(x), float(y))
        return result

    def rectify(self, frame: np.ndarray, width: int | None = None) -> np.ndarray:
        """The frame warped onto the reference map (top-down view), ``width`` pixels wide."""
        import cv2

        w, h = self.reference_size
        scale = 1.0 if not width else width / w
        out_w, out_h = int(round(w * scale)), int(round(h * scale))
        scaling = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
        return cv2.warpPerspective(frame, scaling @ self.homography, (out_w, out_h))


class BoardReference:
    """The canonical board picture and its SIFT features; ``locate`` registers frames to it."""

    def __init__(
        self,
        image_path: Path | None = None,
        *,
        image: np.ndarray | None = None,
        max_features: int = 8000,
        ratio: float = 0.75,
        ransac_px: float = 5.0,
        min_inliers: int = 25,
    ):
        import cv2

        if image is None:
            path = Path(image_path or BOARD_REFERENCE_IMAGE)
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"board reference image not found or unreadable: {path}")
            self.path: Path | None = path
        else:
            self.path = None
        self.image = image
        self.height, self.width = image.shape[:2]
        self.ratio = ratio
        self.ransac_px = ransac_px
        self.min_inliers = min_inliers
        self._sift = cv2.SIFT_create(nfeatures=max_features)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self._sift.detectAndCompute(grey, None)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        log.info("board reference %dx%d: %d features", self.width, self.height, len(self.keypoints))

    def locate(self, frame: np.ndarray, *, scale: float = 1.0) -> Registration | None:
        """Register ``frame`` to the reference; ``scale`` < 1 detects on a downscaled copy (faster)."""
        import cv2

        started = time.perf_counter()
        work = frame
        if scale != 1.0:
            work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
        keypoints, descriptors = self._sift.detectAndCompute(grey, None)
        if descriptors is None or len(keypoints) < 8 or self.descriptors is None:
            return None
        pairs = self._matcher.knnMatch(descriptors, self.descriptors, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < max(8, self.min_inliers // 2):
            return None
        src = np.float32([keypoints[m.queryIdx].pt for m in good]) / scale
        dst = np.float32([self.keypoints[m.trainIdx].pt for m in good])
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None or inliers < self.min_inliers:
            return None
        return Registration(
            homography=homography,
            inliers=inliers,
            matches=len(good),
            seconds=round(time.perf_counter() - started, 3),
            reference_size=(self.width, self.height),
            frame_size=(int(frame.shape[1]), int(frame.shape[0])),
        )


def draw_registration(frame: np.ndarray, registration: Registration) -> np.ndarray:
    """A copy of the frame with the board outline and the space names drawn on it (debugging)."""
    import cv2

    out = frame.copy()
    poly = registration.outline().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [poly], True, (0, 220, 80), 2, cv2.LINE_AA)
    for name, (x, y) in registration.space_pixels().items():
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out, name, (int(x) + 8, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )
    return out


__all__ = ["BoardReference", "Registration", "draw_registration"]
