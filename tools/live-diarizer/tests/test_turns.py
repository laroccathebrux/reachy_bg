from live_diarizer.turns import TurnTracker


def test_consecutive_segments_merge_into_one_turn():
    tracker = TurnTracker(gap=0.3, min_duration=0.2)
    t1 = tracker.update([("speaker0", 0.0, 0.5)], 0.5)
    assert [(t.speaker, t.start, t.end, t.final) for t in t1] == [("speaker0", 0.0, 0.5, False)]
    t2 = tracker.update([("speaker0", 0.5, 1.0)], 1.0)
    assert [(t.start, t.end, t.final) for t in t2] == [(0.0, 1.0, False)]
    t3 = tracker.update([], 1.5)  # 0.5 s of silence > gap: the turn closes
    assert [(t.start, t.end, t.final) for t in t3] == [(0.0, 1.0, True)]
    assert tracker.open == {}


def test_short_gap_does_not_split_and_speakers_are_independent():
    tracker = TurnTracker(gap=0.3, min_duration=0.2)
    tracker.update([("speaker0", 0.0, 0.5)], 0.5)
    tracker.update([("speaker0", 0.5, 0.7), ("speaker1", 0.6, 1.0)], 1.0)
    # speaker0 paused 0.7-0.9 (shorter than the gap) and both talk over each other
    out = tracker.update([("speaker0", 0.9, 1.5), ("speaker1", 1.0, 1.5)], 1.5)
    by = {t.speaker: t for t in out}
    assert by["speaker0"].start == 0.0 and by["speaker0"].end == 1.5 and not by["speaker0"].final
    assert by["speaker1"].start == 0.6 and by["speaker1"].end == 1.5


def test_new_turn_after_long_gap_closes_the_old_one():
    tracker = TurnTracker(gap=0.3, min_duration=0.2)
    tracker.update([("speaker0", 0.0, 1.0)], 1.0)
    out = tracker.update([("speaker0", 2.0, 2.5)], 2.5)
    assert [(t.start, t.end, t.final) for t in out] == [(0.0, 1.0, True), (2.0, 2.5, False)]


def test_tiny_blips_are_dropped():
    tracker = TurnTracker(gap=0.3, min_duration=0.2)
    assert tracker.update([("speaker0", 0.0, 0.1)], 0.5) == []
    assert tracker.update([], 1.0) == []
    assert tracker.flush() == []
