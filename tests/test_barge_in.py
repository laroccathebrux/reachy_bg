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
    asr = FakeTranscriber(["Reachy", ""])
    check = EchoAwareBargeIn(mic, asr, lambda: ANSWER, min_ms=600, recheck_ms=1000)
    mic.voice_ms = 700
    assert check() is False
    mic.voice_ms = 1800
    assert check() is False
    mic.voice_ms = 0
    assert check() is False and check._checked_ms == 0
