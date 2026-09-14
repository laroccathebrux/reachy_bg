"""Text-to-speech with one native voice per language.

ElevenLabs is the default provider. Audio is requested as raw 16 kHz PCM and wrapped into a
WAV file so the caller knows its duration without decoding anything; the robot's
``play_sound`` and macOS ``afplay`` both accept WAV.

    from src.speech.tts import synthesize
    clip = synthesize("Doom advances by one.", "en-US")
    clip.path, clip.duration_s
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import DATA_DIR, ELEVENLABS_API_KEY, ELEVENLABS_MODEL_ID, TTS_PROVIDER, voice_for

SAMPLE_RATE = 16_000
_ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_AUDIO_DIR = DATA_DIR / "tts"


class TTSError(RuntimeError):
    """Raised when speech could not be synthesized."""


@dataclass(frozen=True)
class Clip:
    path: Path
    duration_s: float
    language: str
    voice_id: str
    text: str


def pcm16_to_wav(pcm: bytes, path: Path, sample_rate: int = SAMPLE_RATE) -> float:
    """Write mono 16-bit PCM to ``path`` as WAV; return its duration in seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return len(pcm) / 2 / sample_rate


def elevenlabs_request(
    text: str, voice_id: str, *, model_id: str = ELEVENLABS_MODEL_ID
) -> tuple[str, dict, dict]:
    """(url, params, json body) for a PCM-16k request; split out so it can be unit-tested."""
    url = _ELEVENLABS_URL.format(voice_id=voice_id)
    params = {"output_format": f"pcm_{SAMPLE_RATE}"}
    body = {"text": text, "model_id": model_id}
    return url, params, body


def synthesize(
    text: str,
    language: str,
    *,
    provider: str = TTS_PROVIDER,
    api_key: str = ELEVENLABS_API_KEY,
    out_dir: Path = _AUDIO_DIR,
    timeout: float = 60.0,
) -> Clip:
    """Synthesize ``text`` with the native voice for ``language`` and return the WAV clip."""
    text = " ".join(text.split())
    if not text:
        raise TTSError("nothing to say")
    voice_id = voice_for(language, provider)
    if not voice_id:
        raise TTSError(f"no {provider} voice configured for {language}")
    if provider != "elevenlabs":
        raise TTSError(f"TTS provider {provider!r} is not implemented yet")
    if not api_key:
        raise TTSError("ELEVENLABS_API_KEY is empty")

    url, params, body = elevenlabs_request(text, voice_id)
    try:
        response = httpx.post(url, params=params, json=body, headers={"xi-api-key": api_key}, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise TTSError(f"ElevenLabs request failed: {exc}") from exc

    path = out_dir / f"{int(time.time() * 1000)}_{language}.wav"
    duration = pcm16_to_wav(response.content, path)
    return Clip(path=path, duration_s=duration, language=language, voice_id=voice_id, text=text)


__all__ = ["Clip", "TTSError", "synthesize", "pcm16_to_wav", "elevenlabs_request", "SAMPLE_RATE"]
