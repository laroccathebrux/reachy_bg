"""Runtime configuration for the Reachy Mini Eldritch Horror agent.

Every setting comes from the environment (or a ``.env`` file at the repository root) with a
sensible local default. Nothing here touches the network or the filesystem at import time,
except reading ``.env``; call :func:`ensure_data_dirs` explicitly when a module needs the
runtime data folders.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# REACHY_BG_DOTENV overrides the .env location; an empty value disables loading (tests).
_dotenv = os.getenv("REACHY_BG_DOTENV")
if _dotenv is None:
    load_dotenv(BASE_DIR / ".env")
elif _dotenv:
    load_dotenv(_dotenv)


def _env_str(name: str, default: str = "") -> str:
    """An environment string with a trailing ``# comment`` dropped, the way .env files read.

    ``AUDIO_INPUT_DEVICE=              # empty = system default`` is what the shipped .env.example
    looks like, and the loader keeps that comment as part of the value: the microphone then went
    looking for a device called "# empty = system default input...". A "#" that follows
    whitespace ends the value; one inside a word (a URL fragment, a name) is left alone.
    """
    value = os.getenv(name, default)
    if not value:
        return value
    value = re.split(r"\s+#", value.strip(), maxsplit=1)[0].strip()
    return "" if value.startswith("#") else value  # the whole value was the comment


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
# How long Ollama keeps the reasoning model in memory after a call; a reload costs ~30 s.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

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
AUDIO_INPUT_DEVICE = _env_str("AUDIO_INPUT_DEVICE", "")
AUDIO_SAMPLE_RATE = _env_int("AUDIO_SAMPLE_RATE", 16000)
# Energy-based voice activity detection on the Mac input (src/speech/microphone.py). A frame
# counts as speech when its level exceeds both the absolute floor and the adaptive noise
# floor plus a margin; an utterance ends after VAD_SILENCE_MS of silence.
VAD_FRAME_MS = _env_int("VAD_FRAME_MS", 30)
VAD_SILENCE_MS = _env_int("VAD_SILENCE_MS", 600)
VAD_MIN_SPEECH_MS = _env_int("VAD_MIN_SPEECH_MS", 300)
VAD_PRE_ROLL_MS = _env_int("VAD_PRE_ROLL_MS", 300)
VAD_MAX_UTTERANCE_S = float(os.getenv("VAD_MAX_UTTERANCE_S", "20"))
VAD_THRESHOLD_DBFS = float(os.getenv("VAD_THRESHOLD_DBFS", "-50"))
VAD_NOISE_MARGIN_DB = float(os.getenv("VAD_NOISE_MARGIN_DB", "8"))
# While the robot speaks its own voice reaches the Mac microphone, so interrupting it
# (barge-in) needs a louder, longer voice than a normal utterance start.
# Measured on 2026-09-14 with the MacBook microphone: room floor -48 dBFS, robot voice peaks
# at -25 dBFS, so the bar sits at floor + 8 + 16 = -24 dBFS (a close, raised voice).
# Since the echo-aware check (src/speech/barge_in.py) transcribes what the microphone hears
# and ignores the robot's own words, the margin only has to keep breathing and rustling out.
BARGE_IN_MARGIN_DB = float(os.getenv("BARGE_IN_MARGIN_DB", "4"))
BARGE_IN_MIN_MS = _env_int("BARGE_IN_MIN_MS", 600)
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
# Speaker voiceprints (pyannote/wespeaker-voxceleb-resnet34-LM, 256-d). Cosine similarity
# above the threshold names the speaker; below it the utterance is "unknown". Measured on
# 2026-09-14 with short far-field clips: same person 0.36-0.59, different voices -0.1-0.15.
SPEAKER_EMBEDDING_MODEL = os.getenv("SPEAKER_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM")
SPEAKER_MATCH_THRESHOLD = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.30"))
# The robot's name as players say it, plus what Whisper tends to hear instead.
ROBOT_NAME = os.getenv("ROBOT_NAME", "Reachy")
ROBOT_NAME_ALIASES = tuple(
    a.strip().lower()
    for a in os.getenv(
        "ROBOT_NAME_ALIASES",
        "reachy,reachie,reachi,reaxi,reed,ride,riche,richie,richy,rich,ritch,ritchie,ritchy,reach,richi,ricci,rishi,ricky,rachi",
    ).split(",")
    if a.strip()
)
# A question within this many seconds after the robot finished speaking is a follow-up to it.
FOLLOW_UP_WINDOW_S = float(os.getenv("FOLLOW_UP_WINDOW_S", "8"))
# Once somebody has spoken to the robot, they keep the floor for this long: their next
# sentences are for it too, even when they are statements with no name and no question in them.
# People explain a board in several breaths, with pauses for moving pieces. 30 s is what the
# first live session measured: replaying it (scripts/rules_replay.py) 20 s recovered 7 of the
# owner's dropped instructions, 25 s recovered 8, 30 s recovered all 9, and 40 s added nothing.
FLOOR_WINDOW_S = float(os.getenv("FLOOR_WINDOW_S", "30"))
# Answer rules questions asked to the table without naming the robot (a knowledgeable player
# would); false = only when named, after its own turn, or on its game turn.
ANSWER_GAME_QUESTIONS = _env_bool("ANSWER_GAME_QUESTIONS", True)
# The addressee gate itself. Off means the robot answers everything it hears: no utterance is
# ever dropped, and the rules only watch and log. It still holds each utterance long enough to
# recognise its own echo and to take its own turn, so the robot and the agent never answer the
# same sentence. ``--no-gate`` on the command line does the same for one session.
ADDRESSEE_GATE = _env_bool("ADDRESSEE_GATE", True)
# One language per session, chosen when it starts. The robot answered a Portuguese table in
# English mid-game and the owner stopped the session over it: with the lock on, the agent has no
# tool to switch with, no second voice preset, and a prompt that names the language it speaks.
LANGUAGE_LOCK = _env_bool("LANGUAGE_LOCK", True)
# Hugging Face token: required once to download the gated pyannote models.
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --------------------------------------------------------------------------- voice output
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # "elevenlabs" | "local"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

# --------------------------------------------------------------------------- conversation agent
# The spoken conversation runs on an ElevenLabs agent (ASR + LLM + TTS + turn-taking in the
# cloud, ~0.5 s turns); rules and knowledge stay local as client tools. See docs/SPEECH_PIPELINE.md.
ELEVEN_AGENT_ID = os.getenv("ELEVEN_AGENT_ID", "")  # empty = create once, remember in data/eleven_agent.json
ELEVEN_AGENT_NAME = os.getenv("ELEVEN_AGENT_NAME", "Reachy Eldritch Horror")
# gpt-4.1-mini follows the prompt (game terms in English, language switching) better than the
# platform default (gemini-2.5-flash); empty = platform default.
ELEVEN_AGENT_LLM = os.getenv("ELEVEN_AGENT_LLM", "gpt-4.1-mini")
ELEVEN_TURN_TIMEOUT_S = float(os.getenv("ELEVEN_TURN_TIMEOUT_S", "300"))
ELEVEN_MAX_DURATION_S = _env_int("ELEVEN_MAX_DURATION_S", 7200)
# Audio went to the agent and nothing came back for this long: the cloud session is probably
# dead. The table hears silence and, without this, so does the log - measured against the
# 2026-09-17 session, where a healthy answer came back 1.2 to 2.5 s after the gate released.
# Does the robot think between rounds (src/strategy/reflect.py)? The loop runs on its own
# thread while the humans take their turns and writes priorities the score reads next round. It
# is never in the turn path - "think=True" in the turn was measured at 114 s against 4 s and
# rejected - so switching it off costs turns nothing and only makes them less deliberate.
REFLECT_BETWEEN_ROUNDS = os.getenv("REFLECT_BETWEEN_ROUNDS", "true").lower() not in ("0", "false", "no")

AGENT_QUIET_S = float(os.getenv("AGENT_QUIET_S", "12"))
# How much of an utterance has to stand above its own quiet floor before the "the agent has gone
# silent" clock is started for it (src/speech/microphone.py, voiced_fraction). Whisper writes
# "E aí" over room noise, the gate forwards it, the agent rightly says nothing, and the watchdog
# reported the cloud session dead: 73 of those in one day against 422 real utterances. Measured
# over 218 forwarded clips, 0.60 leaves 2 of 51 noise clips arming the clock and 96 of 167 real
# sentences still arming it - and missing one costs nothing, because the next sentence arms it.
AGENT_QUIET_MIN_VOICE = float(os.getenv("AGENT_QUIET_MIN_VOICE", "0.60"))
# The robot speaker is a USB audio device on the Mac ("Reachy Mini Audio"); streaming to it
# directly is the only audible path for streamed audio on macOS.
AUDIO_OUTPUT_DEVICE = _env_str("AUDIO_OUTPUT_DEVICE", "Reachy Mini Audio")
HEAD_SWAY = _env_bool("HEAD_SWAY", True)


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
# Agents refuse voices with live moderation (Mariana M), so the agent may need its own native
# voice per language; missing entries fall back to the file-TTS voices.
ELEVEN_AGENT_VOICES = {**ELEVENLABS_VOICES, **_voice_map("ELEVEN_AGENT_VOICE_ID")}


def voice_for(language: str, provider: str = TTS_PROVIDER) -> str:
    """Voice id for ``language`` with the active provider, falling back to the default language."""
    voices = ELEVENLABS_VOICES if provider == "elevenlabs" else LOCAL_TTS_VOICES
    return voices.get(language) or voices.get(DEFAULT_LANGUAGE, "")


# --------------------------------------------------------------------------- vision
VISION_URL = os.getenv("VISION_URL", "http://127.0.0.1:8090")
# Rebuild the camera pipeline when frames stop arriving. Off by default, and the reason is
# measured: on 2026-09-17 a rebuild bound to a different 1080p camera on this Mac (there are
# three) and reported success, so the board matcher produced a confident homography over a
# photo of the room and the robot told the table there were no pieces. Frames arriving is not
# evidence of the right camera, and nothing the SDK or the daemon exposes says which one it
# took. The drought is always logged; only the rebuild is gated.
CAMERA_REBUILD = _env_bool("CAMERA_REBUILD", False)
VISION_MODEL = os.getenv("VISION_MODEL", "yolov8s-worldv2.pt")
VISION_MIN_CONFIDENCE = float(os.getenv("VISION_MIN_CONFIDENCE", "0.35"))

# --------------------------------------------------------------------------- directories
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RULEBOOK_DIR = DATA_DIR / "rulebooks"
GAME_LOG_DIR = DATA_DIR / "game_logs"
CAPTURE_DIR = DATA_DIR / "captures"
# Top-down picture of the whole board (the canonical map every camera frame is registered to).
BOARD_REFERENCE_IMAGE = Path(os.getenv("BOARD_REFERENCE_IMAGE", str(DATA_DIR / "imgs" / "World_Map.webp")))
# Crops of the real pieces, one folder per label, photographed by the robot and named by the owner.
GALLERY_DIR = Path(os.getenv("GALLERY_DIR", str(DATA_DIR / "gallery")))
GALLERY_MATCH_THRESHOLD = float(os.getenv("GALLERY_MATCH_THRESHOLD", "0.72"))
AUDIO_CAPTURE_DIR = CAPTURE_DIR / "audio"
SPEAKER_DIR = DATA_DIR / "speakers"
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
    "OLLAMA_KEEP_ALIVE",
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
    "SPEAKER_EMBEDDING_MODEL",
    "SPEAKER_MATCH_THRESHOLD",
    "ROBOT_NAME",
    "ROBOT_NAME_ALIASES",
    "FOLLOW_UP_WINDOW_S",
    "FLOOR_WINDOW_S",
    "ANSWER_GAME_QUESTIONS",
    "ADDRESSEE_GATE",
    "LANGUAGE_LOCK",
    "HF_TOKEN",
    "TTS_PROVIDER",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_MODEL_ID",
    "ELEVENLABS_VOICES",
    "ELEVEN_AGENT_ID",
    "ELEVEN_AGENT_VOICES",
    "ELEVEN_AGENT_NAME",
    "ELEVEN_AGENT_LLM",
    "ELEVEN_TURN_TIMEOUT_S",
    "ELEVEN_MAX_DURATION_S",
    "AGENT_QUIET_S",
    "REFLECT_BETWEEN_ROUNDS",
    "AGENT_QUIET_MIN_VOICE",
    "AUDIO_OUTPUT_DEVICE",
    "HEAD_SWAY",
    "LOCAL_TTS_VOICES",
    "voice_for",
    "VISION_URL",
    "CAMERA_REBUILD",
    "VISION_MODEL",
    "VISION_MIN_CONFIDENCE",
    "DATA_DIR",
    "RULEBOOK_DIR",
    "GAME_LOG_DIR",
    "CAPTURE_DIR",
    "BOARD_REFERENCE_IMAGE",
    "GALLERY_DIR",
    "GALLERY_MATCH_THRESHOLD",
    "AUDIO_CAPTURE_DIR",
    "SPEAKER_DIR",
    "MODEL_DIR",
    "ensure_data_dirs",
    "LOG_LEVEL",
    "LOG_FILE",
    "ENVIRONMENT",
    "DEBUG",
    "validate_config",
]
