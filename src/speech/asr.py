"""Speech recognition with mlx-whisper on the Apple GPU: text, language and confidence.

    transcriber = Transcriber()          # WHISPER_MODEL, WHISPER_LANGUAGE from the config
    transcriber.warm_up()                # load the weights before the first real utterance
    result = transcriber.transcribe(utterance.audio)
    result.text, result.language ("pt-BR" / "en-US"), result.language_confidence, result.seconds

Language identification runs first on the mel spectrogram, restricted to the configured
``SPOKEN_LANGUAGES`` (Whisper otherwise likes to hear Galician or Spanish in Brazilian
Portuguese); decoding then runs with that language pinned. The same ``ModelHolder`` cache as
``mlx_whisper.transcribe`` is used, so the weights are loaded once per process.

The heavy imports happen inside :class:`Transcriber`, so the pure helpers (language mapping,
hallucination filter) are testable without MLX.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import SPOKEN_LANGUAGES, WHISPER_LANGUAGE, WHISPER_MODEL, WHISPER_PROMPT
from src.logger import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16_000

# Whisper decoding thresholds (its own defaults): a segment with a high no-speech probability
# AND a low average log-probability is silence or noise, not words.
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
# Below this probability the language head is guessing; the speaker's last language wins.
LANGUAGE_MIN_CONFIDENCE = 0.5

# Phrases Whisper produces from silence, breathing or music; matched after normalisation.
_HALLUCINATIONS = frozenset(
    {
        "",
        "obrigado",
        "obrigada",
        "obrigado por assistir",
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "legendas pela comunidade amara org",
        "legendas pela comunidade amaraorg",
        "tchau",
        "bye",
        "you",
        "hmm",
        "mm",
    }
)


class ASRError(RuntimeError):
    """Raised when the recognizer cannot run."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str  # BCP-47 from SPOKEN_LANGUAGES, or "" when unsupported / unknown
    whisper_language: str  # Whisper's own two-letter code
    language_confidence: float
    avg_logprob: float
    no_speech_prob: float
    audio_s: float
    seconds: float  # wall-clock latency of the whole call

    @property
    def empty(self) -> bool:
        return not self.text


# --------------------------------------------------------------------------- pure helpers
def map_language(code: str, supported: tuple[str, ...] = SPOKEN_LANGUAGES) -> str:
    """Whisper's ``pt`` / ``en`` -> the configured BCP-47 tag (``pt-BR`` / ``en-US``), or ``""``."""
    primary = (code or "").split("-")[0].lower()
    for tag in supported:
        if tag.split("-")[0].lower() == primary:
            return tag
    return ""


def whisper_code(tag: str) -> str:
    """``pt-BR`` -> ``pt``; ``auto`` and empty stay empty (let Whisper decide)."""
    tag = (tag or "").strip()
    return "" if tag.lower() in ("", "auto") else tag.split("-")[0].lower()


def choose_language(
    probs: dict[str, float], supported: tuple[str, ...] = SPOKEN_LANGUAGES
) -> tuple[str, float]:
    """Most probable *supported* language from Whisper's distribution: ``(tag, probability)``.

    Probability is Whisper's raw value for that language, so a Portuguese speaker heard as
    60 % Portuguese, 30 % Galician and 10 % other yields ``("pt-BR", 0.6)``.
    """
    best_tag, best_prob = "", 0.0
    for tag in supported:
        prob = float(probs.get(whisper_code(tag), 0.0))
        if prob > best_prob:
            best_tag, best_prob = tag, prob
    return best_tag, best_prob


_NORMALISE = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> str:
    return " ".join(_NORMALISE.sub(" ", text.lower()).split())


_NOISE_WORDS = frozenset(
    "cough coughs coughing hmm hm mm mmm ah uh um eh oh laughs laughter music applause".split()
)


def is_hallucination(text: str, avg_logprob: float, no_speech_prob: float) -> bool:
    """True for output that Whisper typically invents on silence, noise, coughs or laughter."""
    normalised = normalise(text)
    if normalised in _HALLUCINATIONS:
        return True
    if normalised and all(word in _NOISE_WORDS for word in normalised.split()):
        return True
    if any(len(word) > 15 and len(set(word)) <= 3 for word in normalised.split()):
        return True  # "Eeeeeeeeee", "hahahahaha": a stretched noise, not speech
    return no_speech_prob > NO_SPEECH_THRESHOLD and avg_logprob < LOGPROB_THRESHOLD


def summarise_segments(segments: list[dict[str, Any]]) -> tuple[float, float]:
    """Duration-weighted average log-probability and the highest no-speech probability."""
    if not segments:
        return LOGPROB_THRESHOLD, 1.0
    total = 0.0
    weighted = 0.0
    no_speech = 0.0
    for seg in segments:
        length = max(0.01, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        total += length
        weighted += length * float(seg.get("avg_logprob", LOGPROB_THRESHOLD))
        no_speech = max(no_speech, float(seg.get("no_speech_prob", 0.0)))
    return weighted / total, no_speech


def to_float32(audio: np.ndarray) -> np.ndarray:
    """int16 (or float) samples -> float32 in [-1, 1], as Whisper expects."""
    if audio.dtype.kind in "iu":
        return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return audio.astype(np.float32)


# --------------------------------------------------------------------------- recognizer
class Transcriber:
    def __init__(
        self,
        model: str = WHISPER_MODEL,
        language: str = WHISPER_LANGUAGE,
        *,
        supported: tuple[str, ...] = SPOKEN_LANGUAGES,
        prompt: str = WHISPER_PROMPT,
    ):
        self.model_name = model
        self.pinned = whisper_code(language)
        self.supported = supported
        self.prompt = prompt or None
        self._model: Any | None = None

    # ------------------------------------------------------------------ model
    def _load(self) -> Any:
        if self._model is None:
            try:
                import mlx.core as mx
                from mlx_whisper.transcribe import ModelHolder
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ASRError("mlx-whisper is not installed; run `uv sync --extra speech`") from exc
            started = time.perf_counter()
            self._model = ModelHolder.get_model(self.model_name, mx.float16)
            log.info("whisper model %s loaded in %.1fs", self.model_name, time.perf_counter() - started)
        return self._model

    def warm_up(self) -> float:
        """Load the weights and run one short decode; returns the seconds it took."""
        started = time.perf_counter()
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.int16))
        return time.perf_counter() - started

    # ------------------------------------------------------------------ recognition
    def identify_language(self, audio: np.ndarray) -> tuple[str, str, float]:
        """``(tag, whisper_code, probability)`` for float32 16 kHz audio."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

        model = self._load()
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(segment)
        distribution = probs[0] if isinstance(probs, list) else probs
        tag, prob = choose_language(distribution, self.supported)
        code = whisper_code(tag) if tag else max(distribution, key=distribution.get)
        return tag, code, prob

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        *,
        fallback_language: str = "",
        min_language_confidence: float = LANGUAGE_MIN_CONFIDENCE,
    ) -> Transcript:
        """Recognize ``audio``; when the language head is unsure, decode in ``fallback_language``.

        Short or cut utterances often get a low-probability wrong language (English at 0.25
        for a Portuguese fragment) and then decode as nonsense; the speaker's previous
        language is the better bet in that case.
        """
        if sample_rate != SAMPLE_RATE:
            from src.speech.microphone import resample

            audio = resample(audio.astype(np.int16), sample_rate, SAMPLE_RATE)
        samples = to_float32(audio)
        audio_s = samples.size / SAMPLE_RATE
        started = time.perf_counter()
        from mlx_whisper.transcribe import transcribe as whisper_transcribe

        self._load()
        if self.pinned:
            tag, code, confidence = map_language(self.pinned, self.supported), self.pinned, 1.0
        else:
            tag, code, confidence = self.identify_language(samples)
            fallback = map_language(fallback_language, self.supported)
            if fallback and confidence < min_language_confidence and tag != fallback:
                log.info("language %s at %.2f is unsure; decoding as %s", tag or code, confidence, fallback)
                tag, code = fallback, whisper_code(fallback)
        try:
            result = whisper_transcribe(
                samples,
                path_or_hf_repo=self.model_name,
                language=code,
                fp16=True,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=self.prompt,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                logprob_threshold=LOGPROB_THRESHOLD,
            )
        except Exception as exc:
            raise ASRError(f"whisper failed: {exc}") from exc
        avg_logprob, no_speech = summarise_segments(result.get("segments", []))
        text = " ".join(str(result.get("text", "")).split())
        if is_hallucination(text, avg_logprob, no_speech):
            text = ""
        seconds = time.perf_counter() - started
        log.info(
            "asr %.2fs for %.1fs audio: %s %.2f | %r",
            seconds,
            audio_s,
            tag or code,
            confidence,
            text,
        )
        return Transcript(
            text=text,
            language=tag,
            whisper_language=code,
            language_confidence=round(confidence, 3),
            avg_logprob=round(avg_logprob, 3) if math.isfinite(avg_logprob) else LOGPROB_THRESHOLD,
            no_speech_prob=round(no_speech, 3),
            audio_s=round(audio_s, 2),
            seconds=round(seconds, 3),
        )


__all__ = [
    "ASRError",
    "LANGUAGE_MIN_CONFIDENCE",
    "Transcript",
    "Transcriber",
    "choose_language",
    "is_hallucination",
    "map_language",
    "normalise",
    "summarise_segments",
    "to_float32",
    "whisper_code",
]
