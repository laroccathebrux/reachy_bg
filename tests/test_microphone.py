"""VAD segmentation, device selection and resampling, all without an audio device."""

import numpy as np
import pytest

from src.speech.microphone import (
    InputDevice,
    MicrophoneError,
    Segmenter,
    list_input_devices,
    read_wav,
    resample,
    resolve_input_device,
    rms_dbfs,
    write_wav,
)

RATE = 16_000
FRAME_MS = 30
FRAME = RATE * FRAME_MS // 1000


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def tone(ms: int, amplitude: float = 0.3, freq: float = 220.0) -> np.ndarray:
    n = RATE * ms // 1000
    t = np.arange(n) / RATE
    return (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)


def noise(ms: int, amplitude: float = 0.002, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = RATE * ms // 1000
    return (rng.standard_normal(n) * amplitude * 32767).astype(np.int16)


def feed(seg: Segmenter, clock: FakeClock, audio: np.ndarray) -> list:
    """Feed ``audio`` frame by frame, advancing the fake clock; return the utterances."""
    out = []
    for start in range(0, audio.size - FRAME + 1, FRAME):
        clock.t += FRAME_MS / 1000
        utt = seg.feed(audio[start : start + FRAME])
        if utt is not None:
            out.append(utt)
    return out


def make_segmenter(clock: FakeClock, **kwargs) -> Segmenter:
    defaults = dict(
        sample_rate=RATE,
        frame_ms=FRAME_MS,
        silence_ms=700,
        min_speech_ms=300,
        pre_roll_ms=300,
        max_utterance_s=5,
        threshold_dbfs=-50,
        noise_margin_db=8,
        clock=clock,
    )
    defaults.update(kwargs)
    return Segmenter(**defaults)


# --------------------------------------------------------------------------- levels
def test_rms_dbfs_scale():
    assert rms_dbfs(np.zeros(FRAME, dtype=np.int16)) < -90
    full = np.full(FRAME, 32767, dtype=np.int16)
    assert abs(rms_dbfs(full)) < 0.01
    half = np.full(FRAME, 16384, dtype=np.int16)
    assert -6.1 < rms_dbfs(half) < -5.9
    assert abs(rms_dbfs(np.full(FRAME, 0.5, dtype=np.float32)) - (-6.02)) < 0.05


# --------------------------------------------------------------------------- segmentation
def test_silence_yields_nothing():
    clock = FakeClock()
    seg = make_segmenter(clock)
    assert feed(seg, clock, noise(3000)) == []
    assert seg.noise_floor_db is not None and seg.noise_floor_db < -50
    assert not seg.in_speech


def test_one_burst_becomes_one_utterance_with_pre_roll_and_tail():
    clock = FakeClock()
    seg = make_segmenter(clock)
    audio = np.concatenate([noise(1000), tone(1000), noise(2000, seed=1)])
    utts = feed(seg, clock, audio)
    assert len(utts) == 1
    utt = utts[0]
    # pre-roll (300 ms) + speech (1 s) + silence tail (700 ms), give or take a frame
    assert 1.9 < utt.duration_s < 2.2
    assert 0.9 < utt.speech_s < 1.1
    assert abs((utt.ended_at - utt.speech_ended_at) - 0.7) < 0.05
    assert -15 < utt.peak_db < -12  # 0.3 amplitude sine: -13.5 dBFS RMS
    assert utt.meta["forced"] is False


def test_two_bursts_become_two_utterances():
    clock = FakeClock()
    seg = make_segmenter(clock)
    audio = np.concatenate([noise(500), tone(800), noise(1000, seed=1), tone(600), noise(1000, seed=2)])
    utts = feed(seg, clock, audio)
    assert len(utts) == 2
    assert utts[0].started_at < utts[0].ended_at <= utts[1].started_at


def test_short_click_is_dropped():
    clock = FakeClock()
    seg = make_segmenter(clock)
    audio = np.concatenate([noise(500), tone(120), noise(1500, seed=1)])
    assert feed(seg, clock, audio) == []


def test_pause_shorter_than_silence_keeps_one_utterance():
    clock = FakeClock()
    seg = make_segmenter(clock)
    audio = np.concatenate([noise(500), tone(500), noise(400, seed=1), tone(500), noise(1000, seed=2)])
    utts = feed(seg, clock, audio)
    assert len(utts) == 1
    assert 1.3 < utts[0].speech_s < 1.5


def test_long_speech_is_cut_at_max_length():
    clock = FakeClock()
    seg = make_segmenter(clock, max_utterance_s=2)
    utts = feed(seg, clock, np.concatenate([noise(300), tone(5000), noise(1000, seed=1)]))
    assert len(utts) >= 2
    assert utts[0].meta["forced"] is True
    assert utts[0].duration_s <= 2.05


def test_flush_returns_speech_in_progress():
    clock = FakeClock()
    seg = make_segmenter(clock)
    assert feed(seg, clock, np.concatenate([noise(300), tone(600)])) == []
    assert seg.in_speech
    utt = seg.flush()
    assert utt is not None and utt.meta["forced"] is True
    assert seg.flush() is None


def test_voice_ms_tracks_the_current_voice():
    clock = FakeClock()
    seg = make_segmenter(clock)
    feed(seg, clock, noise(300))
    assert seg.voice_ms == 0
    feed(seg, clock, tone(600))
    assert 500 <= seg.voice_ms <= 660
    seg.reset()
    assert seg.voice_ms == 0 and not seg.in_speech


def test_extra_margin_ignores_quiet_voice_and_freezes_the_floor():
    clock = FakeClock()
    seg = make_segmenter(clock)
    feed(seg, clock, noise(1000))
    floor = seg.noise_floor_db
    seg.set_extra_margin(30)
    quiet = tone(800, amplitude=0.02)  # about -34 dBFS: speech normally, not under +30 dB
    assert feed(seg, clock, np.concatenate([quiet, noise(1000, seed=1)])) == []
    assert seg.noise_floor_db == floor
    seg.set_extra_margin(0)
    assert len(feed(seg, clock, np.concatenate([quiet, noise(1000, seed=2)]))) == 1


def test_adaptive_floor_rejects_a_louder_but_steady_background():
    clock = FakeClock()
    seg = make_segmenter(clock)
    hum = noise(4000, amplitude=0.05)  # ~ -26 dBFS steady hum, above the absolute floor
    assert feed(seg, clock, hum) == []
    assert seg.threshold_db > -26
    # Speech well above the hum is still detected on top of it.
    assert len(feed(seg, clock, np.concatenate([tone(800), noise(1000, amplitude=0.05, seed=3)]))) == 1


def test_noise_floor_drops_fast_and_rises_slowly():
    clock = FakeClock()
    seg = make_segmenter(clock)
    feed(seg, clock, noise(300, amplitude=0.05))  # starts loud (-26 dBFS)
    feed(seg, clock, noise(2000))  # then quiet (-54 dBFS): the floor follows within seconds
    assert seg.noise_floor_db < -50
    feed(seg, clock, tone(3000))  # a 3 s sentence barely moves it
    assert seg.noise_floor_db < -44


# --------------------------------------------------------------------------- devices
DEVICES = [
    InputDevice(0, "LG ULTRAWIDE", 0, 48000.0),
    InputDevice(4, "HyperX Quadcast", 2, 48000.0),
    InputDevice(5, "Reachy Mini Audio", 2, 16000.0),
    InputDevice(6, "MacBook Pro Microphone", 1, 48000.0),
]


def test_list_input_devices_filters_outputs():
    raw = [
        {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
        {"name": "Mic", "max_input_channels": 1, "default_samplerate": 44100.0},
    ]
    devices = list_input_devices(raw)
    assert devices == [InputDevice(1, "Mic", 1, 44100.0)]


def test_resolve_input_device_by_default_index_substring_and_number():
    assert resolve_input_device("", DEVICES, default_index=4).name == "HyperX Quadcast"
    assert resolve_input_device("", DEVICES, default_index=None).index == 0
    assert resolve_input_device("macbook", DEVICES).index == 6
    assert resolve_input_device("6", DEVICES).name == "MacBook Pro Microphone"
    with pytest.raises(MicrophoneError):
        resolve_input_device("blue yeti", DEVICES)
    with pytest.raises(MicrophoneError):
        resolve_input_device("99", DEVICES)
    with pytest.raises(MicrophoneError):
        resolve_input_device("", [])


# --------------------------------------------------------------------------- resampling / wav
def test_resample_integer_ratio_and_interpolation():
    audio = tone(1000)
    down = resample(np.tile(audio, 3).reshape(3, -1).T.reshape(-1), 48_000, 16_000)
    assert down.size == audio.size
    assert resample(audio, 16_000, 16_000) is audio
    odd = resample(audio, 44_100, 16_000)
    assert abs(odd.size - int(audio.size * 16_000 / 44_100)) <= 1


def test_wav_round_trip(tmp_path):
    audio = tone(500)
    path = write_wav(tmp_path / "a.wav", audio, RATE)
    back = read_wav(path)
    assert back.dtype == np.int16 and back.size == audio.size
    assert np.array_equal(back, audio)
    stereo_48k = np.repeat(np.tile(audio, 3).reshape(3, -1).T.reshape(-1), 2)
    import wave

    with wave.open(str(tmp_path / "s.wav"), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(stereo_48k.astype(np.int16).tobytes())
    mono = read_wav(tmp_path / "s.wav")
    assert mono.size == audio.size
