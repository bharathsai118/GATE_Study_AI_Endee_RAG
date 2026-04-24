"""Ingest GATE CSE study material into Endee."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from backend.embeddings import get_embedding
from backend.endee_client import create_collection, upsert_documents


FOLDER_TYPES = {
    "sample_notes": "notes",
    "previous_year_questions": "pyq",
    "syllabus": "syllabus",
}
EmbeddingFunction = Callable[[str], List[float]]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_header(raw: str) -> Tuple[Dict[str, str], str]:
    parts = raw.split("---", 1)
    if len(parts) == 1:
        return {}, raw.strip()
    metadata: Dict[str, str] = {}
    for line in parts[0].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower().replace(" ", "_")] = value.strip()
    return metadata, parts[1].strip()


def split_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = f"{current[-overlap:]}\n\n{paragraph}".strip()
        else:
            step = max(1, max_chars - overlap)
            chunks.extend(paragraph[start : start + max_chars] for start in range(0, len(paragraph), step))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def split_pyqs(text: str) -> List[str]:
    blocks = [
        block.strip()
        for block in re.split(r"(?=^PYQ\s+\d+)", text, flags=re.MULTILINE)
        if block.strip()
    ]
    return blocks or split_text(text)


def chunk_source(text: str, document_type: str) -> List[str]:
    return split_pyqs(text) if document_type == "pyq" else split_text(text)


def iter_source_files(data_dir: Path = DATA_DIR) -> Iterable[Path]:
    for folder in FOLDER_TYPES:
        source_dir = data_dir / folder
        if source_dir.exists():
            yield from sorted(source_dir.glob("*.txt"))


def extract_chunk_metadata(chunk: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    pyq_match = re.search(r"^PYQ\s+(\d+)", chunk, flags=re.IGNORECASE | re.MULTILINE)
    if pyq_match:
        metadata["pyq_id"] = pyq_match.group(1)
    year_match = re.search(r"year\s*:\s*(\d{4})", chunk, flags=re.IGNORECASE)
    if year_match:
        metadata["year"] = year_match.group(1)
    difficulty_match = re.search(r"difficulty\s*:\s*(easy|medium|hard)", chunk, re.IGNORECASE)
    if difficulty_match:
        metadata["difficulty"] = difficulty_match.group(1).title()
    question_match = re.search(
        r"question\s*:\s*(.+?)(?:\nexpected points:|\Z)",
        chunk,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if question_match:
        metadata["question"] = " ".join(question_match.group(1).split())
    return metadata


def stable_id(path: Path, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{path}|{chunk_index}|{text}".encode("utf-8")).hexdigest()
    return f"gate_chunk_{digest[:16]}"


def build_documents(
    data_dir: Path = DATA_DIR,
    embedding_fn: EmbeddingFunction = get_embedding,
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for path in iter_source_files(data_dir):
        header, body = parse_header(read_text(path))
        document_type = header.get("document_type") or FOLDER_TYPES.get(path.parent.name, "notes")
        base_metadata: Dict[str, Any] = {
            "subject": header.get("subject", "General"),
            "topic": header.get("topic", path.stem.replace("_", " ").title()),
            "difficulty": header.get("difficulty", "Medium"),
            "document_type": document_type,
            "year": header.get("year", ""),
            "source_file": str(path.relative_to(data_dir.parent)).replace("\\", "/"),
        }
        for chunk_index, chunk in enumerate(chunk_source(body, document_type), start=1):
            metadata = dict(base_metadata)
            metadata.update(extract_chunk_metadata(chunk))
            metadata["chunk_index"] = chunk_index
            documents.append(
                {
                    "id": stable_id(path, chunk_index, chunk),
                    "text": chunk,
                    "embedding": embedding_fn(chunk),
                    "metadata": metadata,
                }
            )
    return documents


def ingest_all() -> Dict[str, Any]:
    create_collection()
    documents = build_documents()
    result = upsert_documents(documents)
    return {
        "files_processed": len(list(iter_source_files())),
        "chunks_indexed": len(documents),
        **result,
    }


if __name__ == "__main__":
    stats = ingest_all()
    print(f"Ingested {stats['chunks_indexed']} chunks from {stats['files_processed']} files into Endee.")
