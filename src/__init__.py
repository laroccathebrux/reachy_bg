"""
Eldritch Horror + Reachy Mini: Embodied Conversational AI Agent

Research platform for embodied AI, game-playing agents, and cooperative multi-agent learning.
"""

__version__ = "0.1.0"
__author__ = "Alessandro La Rocca Silveira"
__email__ = "alessandro@example.com"

from .config import (
    QDRANT_HOST,
    QDRANT_PORT,
    OLLAMA_MODEL,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    REACHY_HOST,
    REACHY_PORT,
)
from .logger import get_logger

__all__ = [
    "QDRANT_HOST",
    "QDRANT_PORT",
    "OLLAMA_MODEL",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "REACHY_HOST",
    "REACHY_PORT",
    "get_logger",
]
