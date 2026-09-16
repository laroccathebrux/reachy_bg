"""The spaces of the Eldritch Horror (2013) world map, in reference-image coordinates.

Coordinates are pixels on the canonical board image ``BOARD_REFERENCE_IMAGE`` (a top-down
picture of the board, 1200x790 for the one used to annotate), normalised here to the unit
square so that any other picture of the same board can be used after one rescale. Every
space has a type (city, wilderness, sea) as printed in the board legend; the named spaces
are the nine cities and the six expedition sites, the others carry the printed number.
"""

from __future__ import annotations

from dataclasses import dataclass

ANNOTATED_WIDTH = 1200
ANNOTATED_HEIGHT = 790


@dataclass(frozen=True)
class Space:
    name: str
    kind: str  # city | wilderness | sea
    x: float  # 0..1 of the reference width
    y: float  # 0..1 of the reference height
    radius: float = 0.03  # of the reference width: how far a token can sit from the centre

    def pixel(self, width: int, height: int) -> tuple[float, float]:
        return self.x * width, self.y * height


def _px(name: str, kind: str, x: int, y: int, radius: int = 36) -> Space:
    return Space(name, kind, x / ANNOTATED_WIDTH, y / ANNOTATED_HEIGHT, radius / ANNOTATED_WIDTH)


# Named spaces: the circle (or picture) of the space itself, not its banner.
SPACES: tuple[Space, ...] = (
    _px("San Francisco", "city", 118, 240, 48),
    _px("Arkham", "city", 325, 252, 48),
    _px("London", "city", 527, 205, 48),
    _px("Rome", "city", 598, 300, 48),
    _px("Istanbul", "city", 722, 262, 48),
    _px("Tokyo", "city", 1110, 312, 48),
    _px("Shanghai", "city", 1010, 380, 48),
    _px("Sydney", "city", 1097, 638, 48),
    _px("Buenos Aires", "city", 320, 572, 48),
    _px("The Amazon", "wilderness", 330, 455, 44),
    _px("The Pyramids", "wilderness", 690, 392, 44),
    _px("The Heart of Africa", "wilderness", 665, 512, 44),
    _px("Tunguska", "wilderness", 900, 203, 44),
    _px("The Himalayas", "wilderness", 880, 318, 44),
    _px("Antarctica", "wilderness", 708, 712, 44),
    # Numbered spaces, by the printed number next to the icon.
    _px("1", "city", 73, 180),
    _px("2", "sea", 42, 316),
    _px("3", "sea", 130, 610),
    _px("4", "wilderness", 192, 178),
    _px("5", "city", 216, 240),
    _px("6", "city", 183, 315),
    _px("7", "city", 244, 400),
    _px("8", "sea", 318, 336),  # blue ship-wheel token on the board, as in docs/GAME_REFERENCE.md
    _px("9", "wilderness", 412, 128),
    _px("10", "wilderness", 508, 425),
    _px("11", "sea", 527, 612),
    _px("13", "sea", 695, 100),  # North Atlantic, above London
    _px("12", "sea", 458, 692),  # South Atlantic, below Buenos Aires
    _px("14", "city", 660, 210),
    _px("15", "city", 646, 622),
    _px("16", "city", 806, 210),
    _px("17", "city", 848, 412),
    _px("18", "sea", 823, 652),
    _px("19", "wilderness", 1102, 208),
    _px("20", "city", 1018, 498),
    _px("21", "wilderness", 1057, 566),
)

BY_NAME = {s.name: s for s in SPACES}


def nearest_space(x: float, y: float, *, max_radius_factor: float = 1.6) -> tuple[Space | None, float]:
    """Space whose centre is closest to the normalised point, and the distance in reference widths.

    Returns ``(None, distance)`` when the point is farther than ``max_radius_factor`` times
    the space radius from every centre (a token lying on a path between spaces).
    """
    best, best_d = None, float("inf")
    for space in SPACES:
        d = ((x - space.x) ** 2 + ((y - space.y) * ANNOTATED_HEIGHT / ANNOTATED_WIDTH) ** 2) ** 0.5
        if d < best_d:
            best, best_d = space, d
    if best is not None and best_d > best.radius * max_radius_factor:
        return None, best_d
    return best, best_d


__all__ = ["Space", "SPACES", "BY_NAME", "nearest_space", "ANNOTATED_WIDTH", "ANNOTATED_HEIGHT"]
