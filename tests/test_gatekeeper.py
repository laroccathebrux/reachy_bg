"""The gatekeeper on fake audio, a fake transcriber and synthetic utterances (no microphone)."""

import json
from dataclasses import dataclass, field

import numpy as np

from src.speech.addressee import TurnLogger
from src.speech.gatekeeper import Gatekeeper, NameSpotter
from src.speech.microphone import Utterance


class FakeAudio:
    def __init__(self):
        self.calls = []
        self.last_played_at = 0.0
        self.gate_tail_s = 0.3

    def release_utterance(self, until):
        self.calls.append(("release", until))
        return 8

    def discard_utterance(self, until):
        self.calls.append(("discard", until))
        return 8

    def pass_through(self):
        self.calls.append(("pass", None))
        return 3


@dataclass
class Result:
    text: str
    language: str = "pt-BR"
    language_confidence: float = 0.95


@dataclass
class FakeTranscriber:
    results: list = field(default_factory=list)
    calls: int = 0

    def transcribe(self, audio, sample_rate=16000, **kwargs):
        self.calls += 1
        return self.results.pop(0)


def utterance(start, speech_s=2.0):
    n = int(16000 * (speech_s + 0.6))
    return Utterance(
        audio=np.ones(n, dtype=np.int16),
        sample_rate=16000,
        started_at=start,
        speech_ended_at=start + speech_s,
        ended_at=start + speech_s + 0.6,
        peak_db=-20.0,
    )


def keeper(tmp_path, results, **kwargs):
    audio = FakeAudio()
    asr = FakeTranscriber(list(results))
    clock = {"t": 200.0}
    switches = []
    kw = dict(
        humans=2,
        names=("Alessandro", "Bruno"),
        spoken_recently=lambda: "Você pode fazer duas ações por rodada.",
        robot_spoke_at=lambda: None,
        voice_language=lambda: "pt-BR",
        on_switch=lambda lang, text, source: switches.append((lang, text, source)),
        clock=lambda: clock["t"],
    )
    kw.update(kwargs)
    k = Gatekeeper(audio, asr, TurnLogger(tmp_path / "addressee.jsonl"), **kw)
    return k, audio, asr, clock, switches


def logged(tmp_path):
    return [json.loads(line) for line in (tmp_path / "addressee.jsonl").read_text().splitlines()]


def test_named_utterance_is_released_and_logged_as_live_decision(tmp_path):
    k, audio, asr, clock, _ = keeper(tmp_path, [Result("Reachy, quantas ações eu tenho?")])
    u = utterance(100.0)
    clock["t"] = u.ended_at + 0.9
    verdict = k.judge(u, "Alessandro", 0.7)
    assert verdict.route == "released" and verdict.decision.reason == "name"
    assert audio.calls == [("release", u.ended_at)] and verdict.forwarded == 8
    assert verdict.decision_ms == 900
    record = logged(tmp_path)[0]
    assert record["shadow"] is False and record["route"] == "released" and record["speaker"] == "Alessandro"
    assert k.last_addressed == "Alessandro"


def test_table_talk_and_other_player_named_are_discarded(tmp_path):
    k, audio, asr, _, _ = keeper(
        tmp_path, [Result("Vou viajar para Londres agora"), Result("Bruno, o que você acha?")]
    )
    a = k.judge(utterance(100.0), "Alessandro", 0.6)
    b = k.judge(utterance(110.0), "Alessandro", 0.6)
    assert a.route == "discarded" and a.decision.reason == "not_addressed"
    assert b.route == "discarded" and b.decision.reason == "other_person"
    assert [c[0] for c in audio.calls] == ["discard", "discard"]
    assert all(r["shadow"] is False for r in logged(tmp_path))


def test_rules_question_to_the_table_is_released(tmp_path):
    k, audio, asr, _, _ = keeper(tmp_path, [Result("Quantas ações eu posso fazer por rodada?")])
    verdict = k.judge(utterance(100.0), "Alessandro", 0.6)
    assert verdict.route == "released" and verdict.decision.reason == "game_question"


def test_other_language_switches_the_session_instead_of_releasing_audio(tmp_path):
    k, audio, asr, _, switches = keeper(
        tmp_path, [Result("How many actions can I take per round?", "en-US", 0.97)]
    )
    verdict = k.judge(utterance(100.0), "Alessandro", 0.6)
    assert verdict.route == "switched" and audio.calls == [("discard", 102.6)]
    assert switches == [("en-US", "How many actions can I take per round?", "whisper 0.97")]


def test_echo_is_not_flagged_long_after_playback(tmp_path):
    text = "Quantas ações eu posso fazer por rodada?"  # shares words with the robot's answer
    k, audio, asr, _, _ = keeper(tmp_path, [Result(text), Result(text)])
    audio.last_played_at = 50.0
    late = k.judge(utterance(100.0), "Alessandro", 0.6)
    assert late.decision.reason == "game_question" and late.route == "released"
    audio.last_played_at = 100.2  # playback ended just as this voice began
    close = k.judge(utterance(100.0), "", 0.0)
    assert close.decision.reason == "self_echo" and close.route == "discarded"


def test_voice_fully_inside_playback_is_dropped_without_whisper(tmp_path):
    k, audio, asr, _, _ = keeper(tmp_path, [])
    audio.last_played_at = 103.0
    verdict = k.judge(utterance(100.0), "", 0.0)
    assert verdict.route == "echo_gate" and asr.calls == 0 and audio.calls == [("discard", 102.6)]


def test_empty_transcript_is_dropped(tmp_path):
    k, audio, asr, _, _ = keeper(tmp_path, [Result("")])
    verdict = k.judge(utterance(100.0), "", 0.0)
    assert verdict.route == "no_speech" and audio.calls == [("discard", 102.6)]


def test_inactive_gate_only_watches(tmp_path):
    k, audio, asr, _, switches = keeper(tmp_path, [Result("Vou viajar para Londres agora")], active=False)
    verdict = k.judge(utterance(100.0), "Alessandro", 0.6)
    assert verdict.route == "passed" and audio.calls == []
    assert logged(tmp_path)[0]["shadow"] is True


def test_solo_table_addresses_everything(tmp_path):
    k, audio, asr, _, _ = keeper(
        tmp_path, [Result("Vou viajar para Londres agora")], humans=1, names=("Ana",)
    )
    verdict = k.judge(utterance(100.0), "Ana", 0.6)
    assert verdict.decision.reason == "solo" and verdict.route == "released"


def test_follow_up_only_for_the_player_the_robot_talked_to(tmp_path):
    k, audio, asr, _, _ = keeper(
        tmp_path,
        [Result("Reachy, e a fase de Mythos?"), Result("Sim, claro"), Result("Sim, claro")],
        robot_spoke_at=lambda: 99.0,
    )
    k.judge(utterance(100.0), "Alessandro", 0.7)
    yes = k.judge(utterance(104.0), "Alessandro", 0.7)
    other = k.judge(utterance(105.0), "Bruno", 0.7)
    assert yes.decision.reason == "follow_up_reply" and yes.route == "released"
    assert other.decision.reason == "not_addressed" and other.route == "discarded"


class FakeMic:
    def __init__(self):
        self.voice_ms = 0
        self.audio = np.ones(16000, dtype=np.int16)

    def current_audio(self):
        return self.audio


def test_name_spotter_opens_the_gate_early_and_judge_skips_whisper(tmp_path):
    k, audio, asr, clock, _ = keeper(tmp_path, [Result("Reachy quantas"), Result("unused")])
    mic = FakeMic()
    spot = NameSpotter(mic, k, min_ms=900, recheck_ms=1200, clock=lambda: clock["t"])
    mic.voice_ms = 500
    assert spot() is None and asr.calls == 0
    mic.voice_ms = 1000
    clock["t"] = 101.0
    assert spot() == "Reachy quantas" and audio.calls == [("pass", None)]
    u = utterance(100.0, speech_s=2.5)  # the same voice, closed by the VAD later
    clock["t"] = u.ended_at + 0.1
    verdict = k.judge(u, "Alessandro", 0.7)
    assert verdict.route == "early" and verdict.decision.reason == "name" and asr.calls == 1
    assert k.early_release_at is None
    record = logged(tmp_path)[0]
    assert record["route"] == "early" and record["whisper_ms"] == 0 and record["text"] == "Reachy quantas"


def test_name_spotter_ignores_sentences_without_the_name(tmp_path):
    k, audio, asr, clock, _ = keeper(tmp_path, [Result("vou viajar para Londres")])
    mic = FakeMic()
    spot = NameSpotter(mic, k, min_ms=900, recheck_ms=1200, clock=lambda: clock["t"])
    mic.voice_ms = 1000
    assert spot() is None and audio.calls == [] and asr.calls == 1
    mic.voice_ms = 1500
    assert spot() is None and asr.calls == 1  # not enough new voice for a recheck
