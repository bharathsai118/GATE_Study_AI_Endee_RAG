"""Sentence-transformers embedding utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from backend.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embedding(text: str) -> List[float]:
    """Return a normalized 384-dim embedding for Endee vector search."""

    cleaned = " ".join(text.split()) or "empty text"
    vector = _model().encode(cleaned, normalize_embeddings=True)
    return [float(value) for value in vector.tolist()]
