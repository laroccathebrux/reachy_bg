import numpy as np

from src.speech.barge_in import EchoAwareBargeIn


class FakeMic:
    def __init__(self):
        self.voice_ms = 0
        self.audio = np.zeros(16000, dtype=np.int16)

    def current_audio(self):
        return self.audio


class FakeResult:
    def __init__(self, text):
        self.text = text


class FakeTranscriber:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        return FakeResult(self.texts.pop(0))


ANSWER = "Each investigator may perform up to two actions during the Action Phase."


def test_echo_does_not_interrupt_but_a_player_does():
    mic = FakeMic()
    asr = FakeTranscriber(["investigator may perform up to two actions", "wait, what about Rest?"])
    check = EchoAwareBargeIn(mic, asr, lambda: ANSWER, min_ms=600, recheck_ms=1000)
    mic.voice_ms = 300
    assert check() is False and asr.calls == 0  # too short to bother
    mic.voice_ms = 700
    assert check() is False and asr.calls == 1  # echo
    mic.voice_ms = 1200
    assert check() is False and asr.calls == 1  # not enough new voice since the last check
    mic.voice_ms = 1800
    assert check() is True and asr.calls == 2
    assert check.heard == "wait, what about Rest?"


def test_single_words_and_silence_never_interrupt():
    mic = FakeMic()
    asr = FakeTranscriber(["Reachy stop", ""])
    check = EchoAwareBargeIn(mic, asr, lambda: ANSWER, min_ms=600, recheck_ms=1000)
    mic.voice_ms = 700
    assert check() is False
    mic.voice_ms = 1800
    assert check() is False
    mic.voice_ms = 0
    assert check() is False and check._checked_ms == 0


class FakeRegistry:
    """Voiceprints: the robot's own echo scores near zero, a player well above the threshold."""

    def __init__(self, score, name="Alessandro"):
        self.names = ["Alessandro"]
        self.score = score
        self.name = name
        self.calls = 0

    def identify(self, audio):
        self.calls += 1
        return (self.name if self.score >= 0.3 else ""), self.score


def test_only_the_newest_seconds_are_judged_not_the_whole_segment():
    """The segment opens on the robot's echo and never closes while it speaks, so a check on
    the whole thing is always mostly the robot - which is why nothing ever interrupted."""
    mic = FakeMic()
    mic.audio = np.concatenate([np.full(64000, 1, dtype=np.int16), np.full(32000, 2, dtype=np.int16)])
    seen = []

    class Recorder(FakeTranscriber):
        def transcribe(self, audio, **kwargs):
            seen.append(audio)
            return super().transcribe(audio, **kwargs)

    asr = Recorder(["and what about Rest?"])
    check = EchoAwareBargeIn(mic, asr, lambda: ANSWER, min_ms=600, window_s=2.0, sample_rate=16000)
    mic.voice_ms = 6000
    assert check() is True
    assert seen[0].size == 32000  # the last 2 s only
    assert (seen[0] == 2).all()  # the newest words, not the echo before them


def test_a_known_voice_interrupts_and_the_robots_echo_does_not():
    mic = FakeMic()
    mic.voice_ms = 1000
    asr = FakeTranscriber([])
    player = FakeRegistry(0.68)
    check = EchoAwareBargeIn(mic, asr, lambda: ANSWER, min_ms=600, registry=player)
    assert check() is True
    assert check.speaker == "Alessandro" and check.score == 0.68
    assert asr.calls == 0  # no Whisper on the fast path

    echo = FakeRegistry(0.05)
    check = EchoAwareBargeIn(mic, asr, lambda: ANSWER, min_ms=600, registry=echo)
    assert check() is False and echo.calls == 1


def test_an_empty_registry_falls_back_to_reading_the_words():
    mic = FakeMic()
    mic.voice_ms = 1000
    asr = FakeTranscriber(["and what about Rest?"])
    empty = FakeRegistry(0.9)
    empty.names = []
    check = EchoAwareBargeIn(mic, asr, lambda: ANSWER, min_ms=600, registry=empty)
    assert check() is True and asr.calls == 1 and empty.calls == 0
