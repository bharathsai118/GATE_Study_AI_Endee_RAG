"""Evaluate student answers using Endee-retrieved reference context."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from backend.llm import generate_response
from backend.recommendations import add_attempt
from backend.retrieve import search
from backend.tutor import format_context


STOPWORDS = {"the", "is", "are", "and", "or", "to", "of", "in", "for", "a", "an", "with", "by"}


def _keywords(text: str) -> Set[str]:
    return {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def _parse_score(text: str, default: int) -> int:
    for pattern in (r"score\s*[:\-]\s*(\d{1,2})", r"(\d{1,2})\s*/\s*10"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return max(0, min(10, int(match.group(1))))
    return default


def fallback_evaluation(question: str, answer: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    reference = " ".join(result.get("text", "") for result in results)
    reference_terms = _keywords(reference)
    answer_terms = _keywords(answer)
    overlap = reference_terms & answer_terms
    score = max(1, min(10, round((len(overlap) / max(1, min(len(reference_terms), 25))) * 10)))
    missing = sorted(reference_terms - answer_terms)[:8]
    correct = sorted(overlap)[:8]
    text = (
        f"## Evaluation\nScore: {score}/10\n\n"
        f"### Correct points\n{', '.join(correct) if correct else 'Relevant attempt, but key terms are missing.'}\n\n"
        f"### Missing points\n{', '.join(missing) if missing else 'No major keyword gaps found.'}\n\n"
        f"### Improved answer\nFor '{question}', begin with the direct concept, then add: {reference[:650]}\n\n"
        "### GATE exam tips\n- Use exact terms and formulas.\n- Add complexity or condition when relevant.\n- Keep the answer short and structured."
    )
    return {"evaluation": text, "score": score}


def _usable(value: Any) -> str:
    return "" if value in (None, "", "Any", "All") else str(value)


def _first_metadata(results: List[Dict[str, Any]], key: str) -> str:
    for result in results:
        value = result.get("metadata", {}).get(key)
        if value:
            return str(value)
    return ""


def evaluate_answer(
    question: str,
    student_answer: str,
    filters: Optional[Dict[str, Any]] = None,
    save_progress: bool = True,
) -> Dict[str, Any]:
    results = search(question, filters=filters, top_k=5)
    fallback = fallback_evaluation(question, student_answer, results)
    evaluation = generate_response(
        system_prompt=(
            "Evaluate a GATE CSE answer using only reference context. Return score out of 10, "
            "correct points, missing points, improved answer, and exam tips."
        ),
        user_prompt=(
            f"Question:\n{question}\n\nStudent answer:\n{student_answer}\n\n"
            f"Reference context from Endee:\n{format_context(results)}"
        ),
        fallback_text=fallback["evaluation"],
        temperature=0.1,
    )
    score = _parse_score(evaluation, fallback["score"])
    if save_progress:
        subject = _usable((filters or {}).get("subject")) or _first_metadata(results, "subject") or "General"
        topic = _usable((filters or {}).get("topic")) or _first_metadata(results, "topic") or "General"
        add_attempt("evaluation", subject, topic, score, question)
    return {"evaluation": evaluation, "score": score, "sources": results}
