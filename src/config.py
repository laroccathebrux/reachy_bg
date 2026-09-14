"""
Configuration management for Eldritch Horror Reachy Agent

Load environment variables and provide configuration to application modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# ============================================================================
# QDRANT CONFIGURATION
# ============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

QDRANT_CONFIG = {
    "host": QDRANT_HOST,
    "port": QDRANT_PORT,
    "api_key": QDRANT_API_KEY,
}

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:72b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_CONFIG = {
    "model": OLLAMA_MODEL,
    "base_url": OLLAMA_BASE_URL,
}

# ============================================================================
# ELEVENLABS CONFIGURATION
# ============================================================================

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "default")

ELEVENLABS_CONFIG = {
    "api_key": ELEVENLABS_API_KEY,
    "voice_id": ELEVENLABS_VOICE_ID,
}

# ============================================================================
# REACHY ROBOT CONFIGURATION
# ============================================================================

REACHY_HOST = os.getenv("REACHY_HOST", "localhost")
REACHY_PORT = int(os.getenv("REACHY_PORT", "50051"))
REACHY_CAMERA_PORT = os.getenv("REACHY_CAMERA_PORT", "/dev/ttyUSB0")

REACHY_CONFIG = {
    "host": REACHY_HOST,
    "port": REACHY_PORT,
    "camera_port": REACHY_CAMERA_PORT,
}

# ============================================================================
# APPLICATION DIRECTORIES
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
GAME_LOG_DIR = Path(os.getenv("GAME_LOG_DIR", str(DATA_DIR / "game_logs")))
CAPTURE_DIR = Path(os.getenv("CAPTURE_DIR", str(DATA_DIR / "captures")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(DATA_DIR / "models")))

# Create directories if they don't exist
for directory in [GAME_LOG_DIR, CAPTURE_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = Path(os.getenv("LOG_FILE", str(BASE_DIR / "app.log")))

# ============================================================================
# ENVIRONMENT
# ============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate required configuration"""
    errors = []
    
    if not ELEVENLABS_API_KEY and ENVIRONMENT == "production":
        errors.append("ELEVENLABS_API_KEY is required in production")
    
    if errors:
        raise RuntimeError(f"Configuration errors:\n" + "\n".join(errors))

# Optional: validate on import (comment out if not needed during development)
# validate_config()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Qdrant
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_API_KEY",
    "QDRANT_CONFIG",
    # Ollama
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_CONFIG",
    # ElevenLabs
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "ELEVENLABS_CONFIG",
    # Reachy
    "REACHY_HOST",
    "REACHY_PORT",
    "REACHY_CAMERA_PORT",
    "REACHY_CONFIG",
    # Directories
    "BASE_DIR",
    "DATA_DIR",
    "GAME_LOG_DIR",
    "CAPTURE_DIR",
    "MODEL_DIR",
    # Logging
    "LOG_LEVEL",
    "LOG_FILE",
    # Environment
    "ENVIRONMENT",
    "DEBUG",
]
