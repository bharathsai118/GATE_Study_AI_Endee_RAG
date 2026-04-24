"""Configuration for the GATE Study AI Tutor example."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
PROGRESS_PATH = STORAGE_DIR / "student_progress.json"

ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8080").rstrip("/")
ENDEE_INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "gate_study_ai_tutor")
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "").strip()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_OPENAI_BASE_URL = os.getenv(
    "GEMINI_OPENAI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "140"))


def endee_api_base_url() -> str:
    """Return Endee's HTTP API base URL."""

    if ENDEE_BASE_URL.endswith("/api/v1"):
        return ENDEE_BASE_URL
    return f"{ENDEE_BASE_URL}/api/v1"
