# GATE Study AI Tutor - Advanced Educational and Exam Prep RAG System using Endee

GATE Study AI Tutor is a complete Streamlit example that uses Endee as the vector database for semantic search, retrieval augmented generation, previous-year question discovery, quiz generation, answer evaluation, and weak-topic recommendations for GATE CSE preparation.

This project is designed as an internship submission and as a practical Endee example. It is isolated under `examples/gate_study_ai_tutor` and does not require changes to the core Endee source code.

## Problem Statement

GATE CSE aspirants study from notes, syllabus outlines, and previous-year questions spread across many files. They need topic-wise semantic search, grounded explanations, focused quizzes, answer feedback, and weak-topic revision suggestions. A generic chatbot can hallucinate or ignore the exact study material.

## Solution Overview

The app ingests local GATE CSE notes, PYQs, and syllabus text files. It splits them into chunks, generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and stores the vectors plus metadata in Endee through the Endee HTTP API. Streamlit provides a demo UI for retrieval and learning workflows.

If no OpenAI or Gemini-compatible API key is provided, the app still works in fallback mode by building structured answers from the Endee-retrieved context.

## How Endee Is Used

Endee is the vector database for this project.

- `backend/endee_client.py` creates the Endee index with `POST /api/v1/index/create`.
- `backend/endee_client.py` upserts vectors with `POST /api/v1/index/{index_name}/vector/insert`.
- `backend/endee_client.py` searches vectors with `POST /api/v1/index/{index_name}/search`.
- `backend/retrieve.py` embeds a student query and delegates semantic retrieval to Endee.
- Metadata fields are stored as Endee filter fields for exact filtering by subject, topic, difficulty, document type, year, and source file.

This makes Endee visibly responsible for vector storage and semantic retrieval, not just a dependency listed in the README.

## Features

- Ingest `.txt` files from notes, PYQs, and syllabus folders
- Generate embeddings with `all-MiniLM-L6-v2`
- Store chunk vectors and metadata in Endee
- AI doubt solver with grounded RAG answers
- Semantic search over GATE previous-year questions
- Metadata filtering by subject, topic, and difficulty
- Quiz generator for MCQs and short-answer questions
- Answer evaluator with score, correct points, missing points, improved answer, and GATE tips
- Weak-topic tracker using local JSON progress
- Personalized PYQ recommendations from Endee search
- Fallback mode when no paid LLM key is available

## Architecture

```text
GATE notes / PYQs / syllabus
  |
  v
backend.ingest
  |
  +--> parse metadata headers
  +--> split notes into chunks and PYQs into individual questions
  +--> sentence-transformers embeddings
  +--> Endee HTTP vector insert

Student query
  |
  v
backend.retrieve
  |
  +--> query embedding
  +--> Endee HTTP vector search + metadata filters
  +--> retrieved source chunks
  |
  v
Tutor / Quiz / Evaluator / Recommendations
  |
  +--> LLM generation if key exists
  +--> grounded fallback response otherwise
```

## Folder Structure

```text
examples/gate_study_ai_tutor/
|-- README.md
|-- requirements.txt
|-- .env.example
|-- app.py
|-- backend/
|   |-- config.py
|   |-- endee_client.py
|   |-- embeddings.py
|   |-- ingest.py
|   |-- retrieve.py
|   |-- llm.py
|   |-- tutor.py
|   |-- quiz.py
|   |-- evaluator.py
|   `-- recommendations.py
|-- data/
|   |-- sample_notes/
|   |-- previous_year_questions/
|   `-- syllabus/
|-- storage/
|   `-- student_progress.json
`-- docs/
    `-- system_design.md
```

## Setup

From the Endee repository root:

```bash
cd examples/gate_study_ai_tutor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
cd examples\gate_study_ai_tutor
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

LLM keys are optional. Leave `OPENAI_API_KEY` and `GEMINI_API_KEY` blank to use fallback mode.

## How To Run Endee

Start a local Endee server with Docker:

```bash
docker run \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```

For a detached terminal-friendly run:

```bash
docker run -d \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```

If the container already exists:

```bash
docker start endee-server
```

Endee should be reachable at:

```text
http://localhost:8080
```

The default app config uses:

```text
ENDEE_BASE_URL=http://localhost:8080
ENDEE_INDEX_NAME=gate_study_ai_tutor
```

## Ingest Data

Run ingestion from this example folder:

```bash
python -m backend.ingest
```

The ingestion pipeline creates the Endee index, embeds all sample GATE files, and inserts the vectors through the Endee HTTP API.

## Run Streamlit App

```bash
streamlit run app.py
```

Open the local Streamlit URL, usually:

```text
http://localhost:8501
```

## Demo Flow

1. Start Endee with Docker.
2. Copy `.env.example` to `.env`.
3. Run `python -m backend.ingest`.
4. Run `streamlit run app.py`.
5. Open **AI Doubt Solver** and ask:
   - `Explain 3NF with a simple example and tell me what GATE expects.`
6. Open **PYQ Semantic Search** and search:
   - `Find GATE questions similar to DBMS normalization.`
7. Generate a quiz for `DBMS -> Normalization`.
8. Evaluate an answer about Round Robin scheduling.
9. Save a low quiz score and open **Recommendations** to retrieve similar PYQs for weak topics.

## Validation

Run a syntax check:

```bash
python -m compileall app.py backend
```

The full ingestion check requires Endee to be running:

```bash
python -m backend.ingest
```

## Future Scope

- Add PDF ingestion and OCR for scanned GATE notes
- Add chapter-wise GATE syllabus progress
- Add timed mock-test mode
- Add reranking after Endee retrieval
- Add per-student authentication
- Add charts for score trends
- Add support for multiple exam tracks beyond GATE CSE

## Internship Submission Note

This example demonstrates a production-style educational RAG workflow with clear Endee vector database usage. It avoids committing secrets, includes realistic GATE CSE sample data, supports no-key fallback responses, and keeps all changes isolated inside `examples/gate_study_ai_tutor`.
