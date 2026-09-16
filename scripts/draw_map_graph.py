"""Draw the path table over the board picture, so a person can check it in one look.

    uv run python scripts/draw_map_graph.py            # writes data/captures/map_graph.png
    uv run python scripts/draw_map_graph.py --list     # also prints the paths, space by space

``src/strategy/map_graph.py`` was read off the printed board at high magnification, one region
at a time. That is careful, not infallible, and a wrong path is a move the robot would offer
that the board does not allow. So the table is drawn back over the same picture in the legend's
own colours - Train red, Ship white, Uncharted yellow - and the owner says what is wrong. The
paths that leave one edge of the map and return on the other are drawn as stubs at both edges
and labelled, because a straight line across the picture would be a lie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as FilePath

import cv2
import numpy as np

sys.path.insert(0, str(FilePath(__file__).resolve().parents[1]))

from src.config import BOARD_REFERENCE_IMAGE, CAPTURE_DIR  # noqa: E402
from src.strategy.map_graph import BY_SPACE, PATHS, SHIP, TRAIN, UNCHARTED, unreachable  # noqa: E402
from src.vision.spaces import BY_NAME  # noqa: E402

COLOURS = {  # BGR, following the board legend
    TRAIN: (60, 60, 210),
    SHIP: (250, 250, 250),
    UNCHARTED: (60, 200, 245),
}
DASHES = {TRAIN: (7, 5), UNCHARTED: (3, 7), SHIP: None}


def _point(name: str, width: int, height: int) -> tuple[int, int]:
    space = BY_NAME[name]
    return int(space.x * width), int(space.y * height)


def _draw_line(img, p0, p1, kind: str, thickness: int = 3) -> None:
    colour = COLOURS[kind]
    pattern = DASHES[kind]
    if pattern is None:
        cv2.line(img, p0, p1, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.line(img, p0, p1, colour, thickness, cv2.LINE_AA)
        return
    on, off = pattern
    length = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) or 1
    step = on + off
    for start in range(0, length, step):
        end = min(start + on, length)
        a = (
            int(p0[0] + (p1[0] - p0[0]) * start / length),
            int(p0[1] + (p1[1] - p0[1]) * start / length),
        )
        b = (
            int(p0[0] + (p1[0] - p0[0]) * end / length),
            int(p0[1] + (p1[1] - p0[1]) * end / length),
        )
        cv2.line(img, a, b, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)


def draw(image: np.ndarray) -> np.ndarray:
    """Return a copy of the board picture with every path and space name drawn on it."""
    out = image.copy()
    height, width = out.shape[:2]
    for path in PATHS:
        p0 = _point(path.a, width, height)
        p1 = _point(path.b, width, height)
        if not path.wrap:
            _draw_line(out, p0, p1, path.kind)
            continue
        # A wrapping path leaves the map: draw a stub at each end, towards its own edge.
        for point, edge_x in (
            (p0, 0 if p0[0] < width // 2 else width),
            (p1, 0 if p1[0] < width // 2 else width),
        ):
            stub = (edge_x, point[1])
            _draw_line(out, point, stub, path.kind, thickness=4)
            label = f"{path.a}/{path.b}".replace(" ", "")
            anchor = (12 if edge_x == 0 else width - 150, point[1] - 8)
            cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[path.kind], 1, cv2.LINE_AA
            )
    for name in BY_NAME:
        point = _point(name, width, height)
        degree = len(BY_SPACE.get(name, ()))
        cv2.circle(out, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, point, 5, (255, 255, 255), -1, cv2.LINE_AA)
        text = f"{name} ({degree})"
        cv2.putText(
            out, text, (point[0] + 9, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (point[0] + 9, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def listing() -> str:
    """The same table as text, one line per space, for reading out or checking by hand."""
    lines = []
    for name in BY_NAME:
        paths = BY_SPACE.get(name, ())
        if not paths:
            lines.append(f"{name}: NO PATH")
            continue
        legs = ", ".join(
            f"{p.other(name)} ({p.kind}{', wrap' if p.wrap else ''})"
            for p in sorted(paths, key=lambda p: (p.kind, p.other(name)))
        )
        lines.append(f"{name}: {legs}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(BOARD_REFERENCE_IMAGE), help="board picture to draw on")
    parser.add_argument("--out", default=str(FilePath(CAPTURE_DIR) / "map_graph.png"))
    parser.add_argument("--scale", type=float, default=2.0, help="enlarge before drawing, for legibility")
    parser.add_argument("--list", action="store_true", help="print the paths space by space")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"no board picture at {args.image}")
        return 1
    if args.scale != 1.0:
        image = cv2.resize(image, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    out_path = FilePath(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), draw(image))
    missing = unreachable()
    print(f"{len(PATHS)} paths drawn over {args.image} -> {out_path}")
    if missing:
        print("spaces with no path at all: " + ", ".join(missing))
    if args.list:
        print(listing())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
