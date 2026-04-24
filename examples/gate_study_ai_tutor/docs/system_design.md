# System Design - GATE Study AI Tutor

## Overview

GATE Study AI Tutor is a Streamlit example that demonstrates a full educational RAG pipeline using Endee as the vector database. The design is intentionally simple: Streamlit calls Python backend modules directly, and those modules call the Endee HTTP API for vector storage and retrieval.

## Components

- `app.py`: Streamlit UI with pages for ingestion, doubt solving, PYQ search, quizzes, answer evaluation, recommendations, and project overview.
- `backend/ingest.py`: loads `.txt` files, parses metadata headers, splits content, creates embeddings, and upserts into Endee.
- `backend/embeddings.py`: wraps `sentence-transformers/all-MiniLM-L6-v2`.
- `backend/endee_client.py`: HTTP-only Endee integration for index creation, vector insertion, and vector search.
- `backend/retrieve.py`: embeds queries and calls Endee vector search.
- `backend/tutor.py`: builds grounded RAG answers.
- `backend/quiz.py`: generates practice questions from retrieved context.
- `backend/evaluator.py`: evaluates student answers against retrieved reference chunks.
- `backend/recommendations.py`: tracks weak topics and retrieves similar PYQs.

## Data Flow

```text
Text files
  |
  v
Parse metadata headers
  |
  v
Split notes into chunks and PYQs into individual question blocks
  |
  v
Generate sentence-transformers embeddings
  |
  v
Insert vectors + metadata into Endee

Student query
  |
  v
Generate query embedding
  |
  v
Endee vector search with optional metadata filters
  |
  v
Retrieved source chunks
  |
  v
Tutor answer, quiz, evaluation, or recommendation
```

## Endee Vector Database Role

Endee stores every study chunk as a vector record. Each record contains:

- `id`: stable chunk identifier
- `vector`: dense embedding
- `meta`: returned context fields, including original text
- `filter`: fields used by Endee metadata filtering

The important Endee HTTP calls are implemented in `backend/endee_client.py`:

- `POST /api/v1/index/create`
- `POST /api/v1/index/{index_name}/vector/insert`
- `POST /api/v1/index/{index_name}/search`

## RAG Pipeline

The RAG workflow has two stages:

1. Retrieval:
   - The question is embedded.
   - Endee returns semantically similar chunks.
   - Filters narrow results by subject, topic, difficulty, document type, year, or source file.

2. Generation:
   - Retrieved chunks are formatted as context.
   - The optional LLM is asked to answer only from that context.
   - If no key is configured, fallback mode creates a structured answer from the retrieved chunks.

## Metadata Filtering

Every chunk includes:

- subject
- topic
- difficulty
- document_type
- year
- source_file

PYQ search automatically adds:

```python
{"document_type": "pyq"}
```

This ensures previous-year question search only returns PYQ chunks.

## Recommendation Logic

Quiz and evaluation attempts are saved in `storage/student_progress.json`. The app groups attempts by subject and topic, computes average score, and treats any average below 6 as weak. For each weak topic, it searches Endee for similar PYQs and displays them as revision recommendations.

## Limitations

- The example currently ingests `.txt` files only.
- Endee must be running locally or reachable through `ENDEE_BASE_URL`.
- Fallback generation is deterministic and less fluent than a real LLM.
- The local JSON progress file is suitable for demos, not multi-user production.

## Future Improvements

- Add PDF and OCR ingestion
- Add reranking after Endee retrieval
- Add user accounts and cloud progress storage
- Add GATE mock tests and timed practice
- Add richer dashboards for weak-topic tracking
