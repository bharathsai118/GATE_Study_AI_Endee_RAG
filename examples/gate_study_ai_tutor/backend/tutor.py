"""RAG tutor answer generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.llm import generate_response
from backend.retrieve import search


def format_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        blocks.append(
            f"Source {index}: {metadata.get('source_file', 'unknown')}\n"
            f"Subject: {metadata.get('subject', '')}; Topic: {metadata.get('topic', '')}; "
            f"Difficulty: {metadata.get('difficulty', '')}; Year: {metadata.get('year', '')}\n"
            f"{result.get('text', '')}"
        )
    return "\n\n".join(blocks)


def fallback_answer(question: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No Endee context was retrieved. Start Endee, run ingestion, and try again."
    return (
        "## Grounded GATE Answer\n"
        f"Question: {question}\n\n"
        "Based on the top Endee-retrieved context:\n\n"
        f"{results[0].get('text', '')[:900]}\n\n"
        "## GATE Exam Tips\n"
        "- Start with the formal definition or formula.\n"
        "- Add the key condition that makes the concept valid.\n"
        "- Finish with a small example or complexity/result statement."
    )


def answer_doubt(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    results = search(question, filters=filters, top_k=top_k)
    context = format_context(results)
    answer = generate_response(
        system_prompt=(
            "You are a GATE CSE tutor. Answer only from the retrieved context. "
            "Keep the response precise, exam-oriented, and easy to revise."
        ),
        user_prompt=(
            f"Student question:\n{question}\n\n"
            f"Retrieved context from Endee:\n{context}\n\n"
            "Write a grounded answer with a compact example and GATE tips."
        ),
        fallback_text=fallback_answer(question, results),
    )
    return {"answer": answer, "sources": results}
