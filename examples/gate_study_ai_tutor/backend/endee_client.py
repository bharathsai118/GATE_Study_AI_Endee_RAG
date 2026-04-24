"""HTTP client for using Endee as the vector database.

This example intentionally uses Endee's HTTP API instead of the Python SDK so
the vector database operations are visible and easy to review:

- create_collection() -> POST /index/create
- upsert_documents() -> POST /index/{name}/vector/insert
- search_documents() -> POST /index/{name}/search
"""

from __future__ import annotations

import json
import math
import zlib
from typing import Any, Dict, Iterable, List, Optional

import msgpack
import requests

from backend.config import EMBEDDING_DIMENSION, ENDEE_INDEX_NAME, ENDEE_TOKEN, endee_api_base_url


Metadata = Dict[str, Any]
VectorDocument = Dict[str, Any]


class EndeeHTTPError(RuntimeError):
    """Raised when the local Endee server cannot complete a vector operation."""


def _headers(content_type: str = "application/json") -> Dict[str, str]:
    headers = {"Content-Type": content_type}
    if ENDEE_TOKEN:
        headers["Authorization"] = ENDEE_TOKEN
    return headers


def _zip_json(data: Dict[str, Any]) -> bytes:
    if not data:
        return b""
    return zlib.compress(json.dumps(data, separators=(",", ":")).encode("utf-8"))


def _unzip_json(data: bytes) -> Dict[str, Any]:
    if not data:
        return {}
    return json.loads(zlib.decompress(data).decode("utf-8"))


def _filter_payload(filters: Optional[Metadata]) -> Optional[List[Dict[str, Any]]]:
    if not filters:
        return None

    clauses: List[Dict[str, Any]] = []
    for key, value in filters.items():
        if value in (None, "", "Any", "All"):
            continue
        clauses.append({key: {"$eq": str(value)}})
    return clauses or None


def create_collection(index_name: str = ENDEE_INDEX_NAME) -> Dict[str, Any]:
    """Create the Endee index used by the GATE tutor example."""

    url = f"{endee_api_base_url()}/index/create"
    payload = {
        "index_name": index_name,
        "dim": EMBEDDING_DIMENSION,
        "space_type": "cosine",
        "M": 16,
        "ef_con": 128,
        "checksum": -1,
        "precision": "int8",
    }
    response = requests.post(url, json=payload, headers=_headers(), timeout=30)
    if response.status_code == 200:
        return {"created": True, "index_name": index_name}
    body = response.text.lower()
    if "exist" in body or "already" in body:
        return {"created": False, "index_name": index_name}
    raise EndeeHTTPError(
        f"Endee index creation failed at {url}: {response.status_code} {response.text[:300]}"
    )


def _to_endee_insert_row(doc: VectorDocument) -> List[Any]:
    metadata = dict(doc["metadata"])
    metadata["text"] = doc["text"]
    vector = [float(value) for value in doc["embedding"]]
    norm = math.sqrt(sum(value * value for value in vector))
    filter_fields = {
        "subject": metadata.get("subject", ""),
        "topic": metadata.get("topic", ""),
        "difficulty": metadata.get("difficulty", ""),
        "document_type": metadata.get("document_type", ""),
        "year": str(metadata.get("year", "")),
        "source_file": metadata.get("source_file", ""),
    }
    return [
        str(doc["id"]),
        _zip_json(metadata),
        json.dumps(filter_fields, separators=(",", ":")),
        norm,
        vector,
    ]


def upsert_documents(
    documents: Iterable[VectorDocument],
    index_name: str = ENDEE_INDEX_NAME,
) -> Dict[str, Any]:
    """Store chunk vectors in Endee through the HTTP vector insert endpoint."""

    docs = list(documents)
    if not docs:
        return {"upserted": 0}

    url = f"{endee_api_base_url()}/index/{index_name}/vector/insert"
    total = 0
    for start in range(0, len(docs), 1000):
        rows = [_to_endee_insert_row(doc) for doc in docs[start : start + 1000]]
        body = msgpack.packb(rows, use_bin_type=True, use_single_float=True)
        response = requests.post(
            url,
            data=body,
            headers=_headers("application/msgpack"),
            timeout=60,
        )
        if response.status_code != 200:
            raise EndeeHTTPError(
                f"Endee vector upsert failed at {url}: "
                f"{response.status_code} {response.text[:300]}"
            )
        total += len(rows)
    return {"upserted": total}


def _normalize_search_result(item: Any) -> Dict[str, Any]:
    if isinstance(item, (list, tuple)):
        metadata = _unzip_json(item[2]) if len(item) > 2 else {}
        return {
            "id": item[1] if len(item) > 1 else "",
            "similarity": float(item[0]) if item else 0.0,
            "text": metadata.get("text", ""),
            "metadata": metadata,
        }

    metadata = dict(item.get("meta") or {})
    return {
        "id": item.get("id", ""),
        "similarity": float(item.get("similarity", item.get("score", 0.0))),
        "text": metadata.get("text", ""),
        "metadata": metadata,
    }


def search_documents(
    query_embedding: List[float],
    top_k: int = 5,
    filters: Optional[Metadata] = None,
    index_name: str = ENDEE_INDEX_NAME,
) -> List[Dict[str, Any]]:
    """Run Endee vector search with optional metadata filters.

    Endee is the semantic retrieval engine here: the app sends a query vector
    and Endee returns the nearest stored GATE notes/PYQ chunks.
    """

    url = f"{endee_api_base_url()}/index/{index_name}/search"
    payload: Dict[str, Any] = {
        "vector": [float(value) for value in query_embedding],
        "k": top_k,
        "ef": 128,
        "include_vectors": False,
        "filter_params": {"prefilter_threshold": 10000, "boost_percentage": 0},
    }
    filter_payload = _filter_payload(filters)
    if filter_payload:
        payload["filter"] = json.dumps(filter_payload, separators=(",", ":"))

    response = requests.post(url, json=payload, headers=_headers(), timeout=30)
    if response.status_code != 200:
        raise EndeeHTTPError(
            f"Endee vector search failed at {url}: {response.status_code} {response.text[:300]}"
        )

    try:
        raw = msgpack.unpackb(response.content, raw=False)
    except Exception:
        raw = response.json()
    results = raw.get("results", raw) if isinstance(raw, dict) else raw
    return [_normalize_search_result(item) for item in results[:top_k]]
