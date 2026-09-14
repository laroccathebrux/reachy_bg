"""Turn the per-step annotations of a streaming diarizer into speaker turns (pure logic)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    final: bool = False

    def to_message(self, wall_offset: float) -> dict:
        """JSON-ready dict; ``wall_offset`` maps stream seconds to ``time.time()`` seconds."""
        return {
            "type": "turn",
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "wall_start": round(self.start + wall_offset, 3),
            "wall_end": round(self.end + wall_offset, 3),
            "final": self.final,
        }


@dataclass
class TurnTracker:
    """Merge consecutive segments of the same speaker into turns.

    Feed the ``(speaker, start, end)`` segments of every step with :meth:`update`, which
    returns the turns to publish: ongoing turns (``final=False``, extended in place) and the
    turns that just closed (``final=True``). A gap shorter than ``gap`` seconds between two
    segments of one speaker does not end the turn; a turn shorter than ``min_duration`` is
    dropped silently.
    """

    gap: float = 0.3
    min_duration: float = 0.2
    open: dict[str, Turn] = field(default_factory=dict)

    def update(self, segments: list[tuple[str, float, float]], step_end: float) -> list[Turn]:
        touched: set[str] = set()
        for speaker, start, end in sorted(segments, key=lambda s: s[1]):
            if end <= start:
                continue
            turn = self.open.get(speaker)
            if turn is not None and start - turn.end <= self.gap:
                turn.end = max(turn.end, end)
            else:
                if turn is not None:
                    self._closed.append(self._close(speaker))
                self.open[speaker] = Turn(speaker, start, end)
            touched.add(speaker)
        out: list[Turn] = list(self._closed)
        self._closed = []
        # Speakers silent for longer than the gap (measured up to the end of this step) close.
        for speaker in list(self.open):
            turn = self.open[speaker]
            if speaker not in touched and step_end - turn.end > self.gap:
                closed = self._close(speaker)
                if closed is not None:
                    out.append(closed)
        out.extend(t for s, t in self.open.items() if s in touched)
        return [t for t in out if t.final or t.end - t.start >= self.min_duration]

    def flush(self) -> list[Turn]:
        out = [t for t in (self._close(s) for s in list(self.open)) if t is not None]
        return out

    _closed: list = field(default_factory=list)

    def _close(self, speaker: str) -> Turn | None:
        turn = self.open.pop(speaker)
        if turn.end - turn.start < self.min_duration:
            return None
        turn.final = True
        return turn


def annotation_segments(annotation) -> list[tuple[str, float, float]]:
    """``pyannote.core.Annotation`` -> ``[(speaker, start, end), ...]``."""
    return [
        (str(label), float(seg.start), float(seg.end))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
