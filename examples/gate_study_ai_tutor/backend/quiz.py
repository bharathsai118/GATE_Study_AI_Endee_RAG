"""Quiz generation from retrieved Endee context."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.llm import generate_response
from backend.retrieve import search
from backend.tutor import format_context


def _sentences(results: List[Dict[str, Any]]) -> List[str]:
    text = " ".join(result.get("text", "") for result in results)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]


def fallback_quiz(results: List[Dict[str, Any]], count: int) -> str:
    if not results:
        return "Run ingestion first so Endee can retrieve context for quiz generation."
    facts = _sentences(results) or [results[0].get("text", "Review this topic.")]
    lines = ["## GATE Practice Quiz"]
    for idx in range(1, count + 1):
        fact = facts[(idx - 1) % len(facts)]
        lines.append(
            f"\n### Q{idx}. MCQ\n"
            "Which option is best supported by the retrieved study context?\n\n"
            f"A. {fact}\n"
            "B. The concept is unrelated to GATE CSE.\n"
            "C. The concept has no exam application.\n"
            "D. The concept only applies to non-technical subjects.\n\n"
            "Answer: A"
        )
        if idx % 2 == 0:
            lines.append(f"\n### Short Answer {idx}\nExplain briefly: {fact[:160]}...")
    return "\n".join(lines)


def generate_quiz(subject: str, topic: str, difficulty: str, count: int) -> Dict[str, Any]:
    filters = {"subject": subject, "topic": topic, "difficulty": difficulty}
    results = search(f"{subject} {topic} {difficulty} GATE practice", filters=filters, top_k=8)
    quiz = generate_response(
        system_prompt="Create GATE CSE practice questions that are answerable from context.",
        user_prompt=(
            f"Create {count} questions for {subject} - {topic} ({difficulty}).\n\n"
            f"Endee context:\n{format_context(results)}\n\n"
            "Mix MCQs and short-answer questions. Include answers."
        ),
        fallback_text=fallback_quiz(results, count),
        temperature=0.35,
    )
    return {"quiz": quiz, "sources": results}
