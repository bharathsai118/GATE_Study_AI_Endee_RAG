"""Retrieval helpers for Endee vector search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.embeddings import get_embedding
from backend.endee_client import search_documents


def search(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    query_embedding = get_embedding(query)
    # Endee vector search is used here for semantic retrieval and metadata filtering.
    return search_documents(query_embedding, top_k=top_k, filters=filters)


def search_pyqs(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    pyq_filters = dict(filters or {})
    pyq_filters["document_type"] = "pyq"
    return search(query, filters=pyq_filters, top_k=top_k)


def source_label(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    parts = [
        metadata.get("source_file", "unknown"),
        metadata.get("subject", ""),
        metadata.get("topic", ""),
        str(metadata.get("year", "")),
    ]
    return " | ".join(part for part in parts if part)
