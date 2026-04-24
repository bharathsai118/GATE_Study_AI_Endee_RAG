"""Weak-topic tracking and recommendation logic."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.config import PROGRESS_PATH


def load_progress(path: Path = PROGRESS_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"attempts": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"attempts": []}


def save_progress(progress: Dict[str, Any], path: Path = PROGRESS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def add_attempt(activity: str, subject: str, topic: str, score: int, question: str = "") -> Dict[str, Any]:
    progress = load_progress()
    progress.setdefault("attempts", []).append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "activity": activity,
            "subject": subject,
            "topic": topic,
            "score": int(score),
            "question": question,
        }
    )
    save_progress(progress)
    return progress


def topic_averages(progress: Dict[str, Any]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for attempt in progress.get("attempts", []):
        key = (attempt.get("subject", "General"), attempt.get("topic", "General"))
        buckets.setdefault(key, []).append(int(attempt.get("score", 0)))
    rows = []
    for (subject, topic), scores in buckets.items():
        rows.append(
            {
                "subject": subject,
                "topic": topic,
                "average_score": round(sum(scores) / len(scores), 2),
                "attempts": len(scores),
            }
        )
    return sorted(rows, key=lambda row: row["average_score"])


def weak_topics(threshold: int = 6) -> List[Dict[str, Any]]:
    return [row for row in topic_averages(load_progress()) if row["average_score"] < threshold]


def recommend_for_weak_topics(top_k: int = 3) -> List[Dict[str, Any]]:
    from backend.retrieve import search_pyqs

    recommendations = []
    for weak in weak_topics():
        query = f"GATE {weak['subject']} {weak['topic']} previous year question revision"
        pyqs = search_pyqs(query, filters={"subject": weak["subject"]}, top_k=top_k)
        recommendations.append({"weak_topic": weak, "pyqs": pyqs})
    return recommendations
