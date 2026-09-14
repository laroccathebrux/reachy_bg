"""Retrieval-augmented generation layer: embeddings, vector store, ingestion.

Collections (Qdrant, bge-m3 embeddings, 1024-d cosine):

* ``bg_rules``      chunks of the official English rulebook and reference guide
* ``bg_knowledge``  investigators, Ancient Ones, monsters, gates, official FAQ
* ``bg_sessions``   round-by-round memory of played games
"""

from src.rag.collections import KNOWLEDGE, RULES, SESSIONS, VECTOR_DIM

__all__ = ["RULES", "KNOWLEDGE", "SESSIONS", "VECTOR_DIM"]
