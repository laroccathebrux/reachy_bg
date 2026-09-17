"""The addressee gate on synthetic 250 ms frames: hold, release, discard, pass-through, echo gate."""

import numpy as np

from src.speech.agent_audio import INPUT_BLOCK, MAX_PASS_FRAMES, RobotAudioInterface


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(hold=True):
    clock = Clock()
    audio = RobotAudioInterface("mic", "speaker", clock=clock)
    sent = []
    audio._callback = sent.append
    audio.hold_utterances = hold
    return audio, clock, sent


def frame(value):
    return np.full((INPUT_BLOCK, 1), value, dtype=np.int16)


def feed(audio, clock, values):
    """One frame per value, 250 ms apart; returns the stamp of the last frame."""
    for v in values:
        clock.t += 0.25
        audio._on_input(frame(v), INPUT_BLOCK, None, None)
    return clock.t


def test_held_frames_are_released_in_order_with_a_silent_tail():
    audio, clock, sent = make()
    end = feed(audio, clock, [1, 2, 3])
    assert sent == [] and audio.held_frames == 3 and audio.holding
    feed(audio, clock, [4])  # the next utterance starts before the decision lands
    forwarded = audio.release_utterance(end)
    assert forwarded == 3
    values = [np.frombuffer(b, dtype=np.int16)[0] for b in sent]
    assert values[:3] == [1, 2, 3]
    tail = sent[3:]
    assert len(tail) == 4 and all(np.frombuffer(b, dtype=np.int16).any() == False for b in tail)  # noqa: E712
    assert audio.held_seconds == 0.25  # frame 4 stays held for the next decision
    assert audio.released_frames == 3


def test_discard_drops_only_the_decided_utterance():
    audio, clock, sent = make()
    end = feed(audio, clock, [1, 2])
    feed(audio, clock, [3])
    assert audio.discard_utterance(end) == 2
    assert sent == [] and audio.discarded_frames == 2
    assert audio.release_utterance(clock.t) == 1
    assert np.frombuffer(sent[0], dtype=np.int16)[0] == 3


def test_pass_through_streams_live_until_the_utterance_ends_plus_tail():
    audio, clock, sent = make()
    feed(audio, clock, [1, 2])
    assert audio.pass_through() == 2  # what was held goes out at once
    feed(audio, clock, [3])
    assert len(sent) == 3 and not audio.holding
    audio.utterance_ended()
    feed(audio, clock, [0, 0, 0, 0])  # 1 s tail passes live
    assert len(sent) == 7 and audio.holding
    feed(audio, clock, [5])
    assert len(sent) == 7 and audio.held_frames == 3


def test_gate_off_passes_everything():
    audio, clock, sent = make(hold=False)
    feed(audio, clock, [1, 2])
    assert len(sent) == 2 and not audio.holding
    assert audio.pass_through() == 0 and audio.release_utterance(clock.t) == 0


def test_echo_gate_holds_first_and_barge_in_release_opens_the_addressee_gate():
    audio, clock, sent = make()
    audio.speaking = True  # the robot's voice is playing
    feed(audio, clock, [7, 8, 9])
    assert sent == [] and audio.gated_frames == 3 and audio.held_frames == 0
    assert audio.release_gate(frames=2) == 2
    assert [np.frombuffer(b, dtype=np.int16)[0] for b in sent] == [8, 9]
    audio.speaking = False
    feed(audio, clock, [10])  # the player's sentence keeps streaming live
    assert len(sent) == 3 and not audio.holding
    audio.utterance_ended()
    feed(audio, clock, [0, 0, 0, 0, 11])
    assert len(sent) == 7 and audio.held_frames == 1


def test_muted_interface_sends_nothing():
    audio, clock, sent = make()
    audio.muted = True
    feed(audio, clock, [1])
    assert audio.release_utterance(clock.t) == 0 and sent == []


def test_a_barge_in_after_the_voice_ended_does_not_leave_the_gate_open():
    """The failure of 2026-09-17: the check that fired the barge-in finished after the local
    VAD had already closed the voice, so utterance_ended() had nothing left to close - the gate
    stayed open, the next sentence streamed live instead of being held, and the release that
    should have closed the agent's turn forwarded nothing."""
    audio, clock, sent = make()
    audio.speaking = True
    feed(audio, clock, [1, 2])
    audio.speaking = False
    audio.release_gate(frames=2, voice_in_progress=False)  # the voice is already over
    assert not audio.holding  # the held frames go out and the tail is armed
    feed(audio, clock, [0, 0, 0, 0])
    assert audio.holding  # the gate closed itself

    end = feed(audio, clock, [3, 4, 5])  # the player's next sentence is held, not streamed
    assert audio.held_frames == 3
    assert audio.release_utterance(end) == 3


def test_an_open_gate_closes_itself_if_the_utterance_never_ends():
    audio, clock, sent = make()
    audio.pass_through()  # a voice was running, so the tail waits for utterance_ended()
    feed(audio, clock, [1] * MAX_PASS_FRAMES)
    assert not audio.holding
    feed(audio, clock, [0, 0, 0, 0])  # the watchdog armed the tail at MAX_PASS_FRAMES
    assert audio.holding


def test_the_turn_closing_silence_goes_out_even_when_nothing_was_held():
    """Holding the microphone again is not silence to the agent - it is no audio at all, and
    its turn detector waits on that for ever. The tail has to go out regardless."""
    audio, clock, sent = make()
    audio.pass_through()  # the frames stream live, so the utterance buffer stays empty
    feed(audio, clock, [1, 2])
    assert audio.release_utterance(clock.t) == 0
    tail = sent[2:]
    assert len(tail) == 4 and all(not np.frombuffer(b, dtype=np.int16).any() for b in tail)
