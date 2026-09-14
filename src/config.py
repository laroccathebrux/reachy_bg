"""Runtime configuration for the Reachy Mini Eldritch Horror agent.

Every setting comes from the environment (or a ``.env`` file at the repository root) with a
sensible local default. Nothing here touches the network or the filesystem at import time,
except reading ``.env``; call :func:`ensure_data_dirs` explicitly when a module needs the
runtime data folders.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# REACHY_BG_DOTENV overrides the .env location; an empty value disables loading (tests).
_dotenv = os.getenv("REACHY_BG_DOTENV")
if _dotenv is None:
    load_dotenv(BASE_DIR / ".env")
elif _dotenv:
    load_dotenv(_dotenv)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- game
GAME_ID = os.getenv("GAME_ID", "eldritch-horror")

# --------------------------------------------------------------------------- languages
# The table is bilingual: players switch between Brazilian Portuguese and English at will.
# Each utterance is language-identified, the robot answers in that language, and every
# language has its own NATIVE voice (a Brazilian voice never reads English and vice versa).
# Code, prompts and data stay in English regardless.
SPOKEN_LANGUAGES = tuple(
    lang.strip() for lang in os.getenv("SPOKEN_LANGUAGES", "pt-BR,en-US").split(",") if lang.strip()
)
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", SPOKEN_LANGUAGES[0] if SPOKEN_LANGUAGES else "en-US")
# How many consecutive utterances in another supported language before the robot switches;
# 1 = switch immediately, 2 = ignore a single stray sentence.
LANGUAGE_SWITCH_THRESHOLD = _env_int("LANGUAGE_SWITCH_THRESHOLD", 1)

# --------------------------------------------------------------------------- Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# --------------------------------------------------------------------------- Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# Reasoning model: Qwen 3.6 35B-A3B (MoE, ~3B active parameters, ~22 GB on disk).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.6:35b-mlx")
# Embedding model shared by every Qdrant collection (1024-d).
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# --------------------------------------------------------------------------- Reachy Mini
# The reachy-mini daemon exposes REST + WebSocket on this host/port (Lite = same machine).
REACHY_HOST = os.getenv("REACHY_HOST", "127.0.0.1")
REACHY_PORT = _env_int("REACHY_PORT", 8000)
# "auto" tries localhost first, then the network host (Wireless model).
REACHY_CONNECTION_MODE = os.getenv("REACHY_CONNECTION_MODE", "auto")
# "default" | "local" | "webrtc" | "no_media". Camera and speaker come from the robot;
# the microphone never does (see AUDIO_INPUT_DEVICE).
REACHY_MEDIA_BACKEND = os.getenv("REACHY_MEDIA_BACKEND", "default")

# --------------------------------------------------------------------------- audio
# The robot's microphone flat cable is broken, so speech input always comes from a Mac
# input device. Empty string = the system default input device (sounddevice semantics).
AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE", "")
AUDIO_SAMPLE_RATE = _env_int("AUDIO_SAMPLE_RATE", 16000)
# Energy-based voice activity detection on the Mac input (src/speech/microphone.py). A frame
# counts as speech when its level exceeds both the absolute floor and the adaptive noise
# floor plus a margin; an utterance ends after VAD_SILENCE_MS of silence.
VAD_FRAME_MS = _env_int("VAD_FRAME_MS", 30)
VAD_SILENCE_MS = _env_int("VAD_SILENCE_MS", 700)
VAD_MIN_SPEECH_MS = _env_int("VAD_MIN_SPEECH_MS", 300)
VAD_PRE_ROLL_MS = _env_int("VAD_PRE_ROLL_MS", 300)
VAD_MAX_UTTERANCE_S = float(os.getenv("VAD_MAX_UTTERANCE_S", "20"))
VAD_THRESHOLD_DBFS = float(os.getenv("VAD_THRESHOLD_DBFS", "-50"))
VAD_NOISE_MARGIN_DB = float(os.getenv("VAD_NOISE_MARGIN_DB", "8"))
# While the robot speaks its own voice reaches the Mac microphone, so interrupting it
# (barge-in) needs a louder, longer voice than a normal utterance start.
# Measured on 2026-09-14 with the MacBook microphone: room floor -48 dBFS, robot voice peaks
# at -25 dBFS, so the bar sits at floor + 8 + 16 = -24 dBFS (a close, raised voice).
BARGE_IN_MARGIN_DB = float(os.getenv("BARGE_IN_MARGIN_DB", "16"))
BARGE_IN_MIN_MS = _env_int("BARGE_IN_MIN_MS", 500)
# Keep every captured utterance as a WAV file for debugging and for the research dataset.
SAVE_CAPTURES = _env_bool("SAVE_CAPTURES", True)

# --------------------------------------------------------------------------- speech
# Whisper checkpoint for mlx-whisper (Metal). "large-v3-turbo" balances Portuguese and
# English quality against latency on an M1 Max. Language is identified per utterance
# ("auto"); set a BCP-47 code to pin it.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
# Optional vocabulary hint for Whisper (game terms it would otherwise misspell).
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "Eldritch Horror, Ancient One, Doom, Omen, Mythos, Clue, Gate, Action Phase, Encounter Phase.",
)
# Live diarizer sidecar (diart in its own venv) publishes speaker events here.
DIARIZER_URL = os.getenv("DIARIZER_URL", "ws://127.0.0.1:8765")
# Hugging Face token: required once to download the gated pyannote models.
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --------------------------------------------------------------------------- voice output
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # "elevenlabs" | "local"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")


def _voice_map(prefix: str) -> dict[str, str]:
    """``{"pt-BR": <voice>, "en-US": <voice>}`` from ``<prefix>_PT_BR`` / ``<prefix>_EN_US`` style vars."""
    voices: dict[str, str] = {}
    for lang in SPOKEN_LANGUAGES:
        value = os.getenv(f"{prefix}_{lang.replace('-', '_').upper()}", "")
        if value:
            voices[lang] = value
    return voices


# One native voice per language. Missing entries fall back to DEFAULT_LANGUAGE's voice.
ELEVENLABS_VOICES = _voice_map("ELEVENLABS_VOICE_ID")
# Local TTS (Piper-style voice names, e.g. pt_BR-faber-medium / en_US-lessac-medium).
LOCAL_TTS_VOICES = _voice_map("LOCAL_TTS_VOICE")


def voice_for(language: str, provider: str = TTS_PROVIDER) -> str:
    """Voice id for ``language`` with the active provider, falling back to the default language."""
    voices = ELEVENLABS_VOICES if provider == "elevenlabs" else LOCAL_TTS_VOICES
    return voices.get(language) or voices.get(DEFAULT_LANGUAGE, "")


# --------------------------------------------------------------------------- vision
VISION_URL = os.getenv("VISION_URL", "http://127.0.0.1:8090")
VISION_MODEL = os.getenv("VISION_MODEL", "yolov8s-worldv2.pt")
VISION_MIN_CONFIDENCE = float(os.getenv("VISION_MIN_CONFIDENCE", "0.35"))

# --------------------------------------------------------------------------- directories
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RULEBOOK_DIR = DATA_DIR / "rulebooks"
GAME_LOG_DIR = DATA_DIR / "game_logs"
CAPTURE_DIR = DATA_DIR / "captures"
AUDIO_CAPTURE_DIR = CAPTURE_DIR / "audio"
MODEL_DIR = DATA_DIR / "models"


def ensure_data_dirs() -> None:
    """Create the runtime data folders (idempotent)."""
    for directory in (RULEBOOK_DIR, GAME_LOG_DIR, CAPTURE_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- logging / env
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = Path(os.getenv("LOG_FILE", str(BASE_DIR / "app.log")))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = _env_bool("DEBUG", False)


def validate_config() -> list[str]:
    """Return human-readable problems with the current configuration (empty = fine)."""
    problems: list[str] = []
    if TTS_PROVIDER == "elevenlabs" and not ELEVENLABS_API_KEY:
        problems.append("TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is empty")
    if DEFAULT_LANGUAGE not in SPOKEN_LANGUAGES:
        problems.append(f"DEFAULT_LANGUAGE={DEFAULT_LANGUAGE} is not in SPOKEN_LANGUAGES={SPOKEN_LANGUAGES}")
    configured = ELEVENLABS_VOICES if TTS_PROVIDER == "elevenlabs" else LOCAL_TTS_VOICES
    for lang in SPOKEN_LANGUAGES:
        if lang not in configured:
            # voice_for() would fall back to another language's voice, which is exactly the
            # "Brazilian voice reading English" problem; surface it instead of hiding it.
            problems.append(f"no native {TTS_PROVIDER} voice configured for {lang}")
    if AUDIO_SAMPLE_RATE not in (16000, 24000, 44100, 48000):
        problems.append(f"AUDIO_SAMPLE_RATE={AUDIO_SAMPLE_RATE} is unusual")
    return problems


__all__ = [
    "BASE_DIR",
    "GAME_ID",
    "SPOKEN_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "LANGUAGE_SWITCH_THRESHOLD",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "EMBED_MODEL",
    "REACHY_HOST",
    "REACHY_PORT",
    "REACHY_CONNECTION_MODE",
    "REACHY_MEDIA_BACKEND",
    "AUDIO_INPUT_DEVICE",
    "AUDIO_SAMPLE_RATE",
    "VAD_FRAME_MS",
    "VAD_SILENCE_MS",
    "VAD_MIN_SPEECH_MS",
    "VAD_PRE_ROLL_MS",
    "VAD_MAX_UTTERANCE_S",
    "VAD_THRESHOLD_DBFS",
    "VAD_NOISE_MARGIN_DB",
    "BARGE_IN_MARGIN_DB",
    "BARGE_IN_MIN_MS",
    "SAVE_CAPTURES",
    "WHISPER_MODEL",
    "WHISPER_LANGUAGE",
    "WHISPER_PROMPT",
    "DIARIZER_URL",
    "HF_TOKEN",
    "TTS_PROVIDER",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_MODEL_ID",
    "ELEVENLABS_VOICES",
    "LOCAL_TTS_VOICES",
    "voice_for",
    "VISION_URL",
    "VISION_MODEL",
    "VISION_MIN_CONFIDENCE",
    "DATA_DIR",
    "RULEBOOK_DIR",
    "GAME_LOG_DIR",
    "CAPTURE_DIR",
    "AUDIO_CAPTURE_DIR",
    "MODEL_DIR",
    "ensure_data_dirs",
    "LOG_LEVEL",
    "LOG_FILE",
    "ENVIRONMENT",
    "DEBUG",
    "validate_config",
]
