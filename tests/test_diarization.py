from src.speech.diarization import SpeakerTurn, TurnLog, dominant_speaker, overlap


def test_overlap():
    assert overlap(0, 2, 1, 3) == 1
    assert overlap(0, 1, 2, 3) == 0


def test_dominant_speaker_by_overlap():
    turns = [SpeakerTurn("speaker0", 10.0, 12.0, True), SpeakerTurn("speaker1", 11.5, 15.0, True)]
    assert dominant_speaker(turns, 10.0, 12.0) == ("speaker0", 2.0)
    assert dominant_speaker(turns, 12.5, 14.0) == ("speaker1", 1.5)
    assert dominant_speaker(turns, 20.0, 21.0) == ("", 0.0)


def test_turn_log_updates_in_place_and_tracks_active():
    log = TurnLog(keep_s=100)
    log.apply({"type": "turn", "speaker": "speaker0", "wall_start": 1.0, "wall_end": 1.5, "final": False})
    log.apply({"type": "turn", "speaker": "speaker0", "wall_start": 1.0, "wall_end": 3.0, "final": True})
    log.apply({"type": "step", "time": 3.5, "active": ["speaker1"]})
    assert [(t.wall_start, t.wall_end, t.final) for t in log.turns()] == [(1.0, 3.0, True)]
    assert log.active == ["speaker1"]
    assert log.speaker_between(0.0, 5.0) == ("speaker0", 2.0)
    log.apply({"type": "turn", "speaker": "speaker1", "wall_start": 500.0, "wall_end": 501.0, "final": True})
    assert [t.speaker for t in log.turns()] == ["speaker1"]  # the old turn was pruned
