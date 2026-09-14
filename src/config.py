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
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- game
GAME_ID = os.getenv("GAME_ID", "eldritch-horror")
# Language the robot listens to and speaks (BCP-47). Code and data stay in English regardless.
SPOKEN_LANGUAGE = os.getenv("SPOKEN_LANGUAGE", "pt-BR")

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

# --------------------------------------------------------------------------- speech
# Whisper checkpoint for mlx-whisper (Metal). "large-v3-turbo" balances Portuguese quality
# and latency on an M1 Max.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
# Live diarizer sidecar (diart in its own venv) publishes speaker events here.
DIARIZER_URL = os.getenv("DIARIZER_URL", "ws://127.0.0.1:8765")
# Hugging Face token: required once to download the gated pyannote models.
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --------------------------------------------------------------------------- voice output
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # "elevenlabs" | "local"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# --------------------------------------------------------------------------- vision
VISION_URL = os.getenv("VISION_URL", "http://127.0.0.1:8090")
VISION_MODEL = os.getenv("VISION_MODEL", "yolov8s-worldv2.pt")
VISION_MIN_CONFIDENCE = float(os.getenv("VISION_MIN_CONFIDENCE", "0.35"))

# --------------------------------------------------------------------------- directories
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RULEBOOK_DIR = DATA_DIR / "rulebooks"
GAME_LOG_DIR = DATA_DIR / "game_logs"
CAPTURE_DIR = DATA_DIR / "captures"
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
    if AUDIO_SAMPLE_RATE not in (16000, 24000, 44100, 48000):
        problems.append(f"AUDIO_SAMPLE_RATE={AUDIO_SAMPLE_RATE} is unusual")
    return problems


__all__ = [
    "BASE_DIR",
    "GAME_ID",
    "SPOKEN_LANGUAGE",
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
    "WHISPER_MODEL",
    "DIARIZER_URL",
    "HF_TOKEN",
    "TTS_PROVIDER",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "VISION_URL",
    "VISION_MODEL",
    "VISION_MIN_CONFIDENCE",
    "DATA_DIR",
    "RULEBOOK_DIR",
    "GAME_LOG_DIR",
    "CAPTURE_DIR",
    "MODEL_DIR",
    "ensure_data_dirs",
    "LOG_LEVEL",
    "LOG_FILE",
    "ENVIRONMENT",
    "DEBUG",
    "validate_config",
]
