"""Streamlit app for the GATE Study AI Tutor example."""

from __future__ import annotations

import streamlit as st

from backend.evaluator import evaluate_answer
from backend.endee_client import EndeeHTTPError
from backend.ingest import ingest_all
from backend.quiz import generate_quiz
from backend.recommendations import (
    add_attempt,
    load_progress,
    recommend_for_weak_topics,
    topic_averages,
)
from backend.retrieve import search_pyqs, source_label
from backend.tutor import answer_doubt


SUBJECTS = ["Any", "DBMS", "Data Structures", "Operating Systems", "Computer Networks"]
TOPICS = [
    "Any",
    "Normalization",
    "Binary Search",
    "Process Scheduling",
    "TCP/IP",
    "GATE CSE Syllabus",
]
DIFFICULTIES = ["Any", "Easy", "Medium", "Hard"]


def clean_filters(filters: dict) -> dict:
    return {key: value for key, value in filters.items() if value not in ("Any", "", None)}


def filter_controls(prefix: str) -> dict:
    return clean_filters(
        {
            "subject": st.selectbox("Subject filter", SUBJECTS, key=f"{prefix}_subject"),
            "topic": st.selectbox("Topic filter", TOPICS, key=f"{prefix}_topic"),
            "difficulty": st.selectbox("Difficulty filter", DIFFICULTIES, key=f"{prefix}_difficulty"),
        }
    )


def safe_run(fn):
    try:
        return fn()
    except EndeeHTTPError as exc:
        st.error(str(exc))
        st.info("Start Endee and run ingestion before retrieval workflows.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


def display_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No Endee source chunks returned.")
        return
    with st.expander("Retrieved Endee source chunks", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            st.markdown(f"**Source {index}: {source_label(source)}**")
            st.caption(
                f"Similarity: {source.get('similarity', 0):.4f} | "
                f"Difficulty: {metadata.get('difficulty', '')}"
            )
            st.write(source.get("text", ""))
            st.divider()


def page_ingest() -> None:
    st.header("Ingest Data")
    st.write("Create an Endee index and store embedded GATE notes, PYQs, and syllabus chunks.")
    st.caption("Default Endee endpoint: http://localhost:8080/api/v1")
    if st.button("Ingest sample GATE data into Endee", type="primary"):
        with st.spinner("Embedding text and sending vectors to Endee HTTP API..."):
            stats = safe_run(ingest_all)
        if stats:
            st.success(f"Indexed {stats['chunks_indexed']} chunks from {stats['files_processed']} files.")
            st.json(stats)


def page_doubt_solver() -> None:
    st.header("AI Doubt Solver")
    filters = filter_controls("doubt")
    question = st.text_area(
        "Ask a GATE CSE question",
        "Explain 3NF with a simple example and tell me what GATE expects.",
        height=120,
    )
    top_k = st.slider("Number of Endee sources", 2, 8, 5)
    if st.button("Generate grounded answer", type="primary"):
        with st.spinner("Searching Endee and generating answer..."):
            result = safe_run(lambda: answer_doubt(question, filters=filters, top_k=top_k))
        if result:
            st.markdown(result["answer"])
            display_sources(result["sources"])


def page_pyq_search() -> None:
    st.header("PYQ Semantic Search")
    filters = filter_controls("pyq")
    query = st.text_input("Topic or question", "Find GATE questions similar to DBMS normalization.")
    top_k = st.slider("PYQ matches", 3, 10, 5)
    if st.button("Search previous-year questions", type="primary"):
        with st.spinner("Running Endee semantic search over PYQ vectors..."):
            results = safe_run(lambda: search_pyqs(query, filters=filters, top_k=top_k))
        if results:
            for item in results:
                metadata = item.get("metadata", {})
                st.subheader(metadata.get("question") or metadata.get("topic", "GATE PYQ"))
                st.caption(
                    f"{metadata.get('subject', '')} | {metadata.get('topic', '')} | "
                    f"Year: {metadata.get('year', 'N/A')} | Source: {metadata.get('source_file', '')}"
                )
                st.write(item.get("text", ""))
                st.divider()
        elif results is not None:
            st.info("No PYQs found. Try fewer filters or ingest the data first.")


def page_quiz() -> None:
    st.header("Quiz Generator")
    topic_map = {
        "DBMS": ["Normalization"],
        "Data Structures": ["Binary Search"],
        "Operating Systems": ["Process Scheduling"],
        "Computer Networks": ["TCP/IP"],
    }
    subject = st.selectbox("Subject", list(topic_map))
    topic = st.selectbox("Topic", topic_map[subject])
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    count = st.number_input("Number of questions", min_value=2, max_value=10, value=4)

    if st.button("Generate quiz from Endee context", type="primary"):
        with st.spinner("Retrieving context from Endee and building quiz..."):
            result = safe_run(lambda: generate_quiz(subject, topic, difficulty, int(count)))
        if result:
            st.markdown(result["quiz"])
            display_sources(result["sources"])

    st.subheader("Save quiz score")
    score = st.slider("Score out of 10", 0, 10, 6)
    if st.button("Save quiz score"):
        add_attempt("quiz", subject, topic, score, "Self-scored generated quiz")
        st.success("Saved score locally for recommendations.")


def page_evaluator() -> None:
    st.header("Answer Evaluator")
    filters = filter_controls("eval")
    question = st.text_area(
        "Question",
        "What is Round Robin scheduling and how does time quantum affect performance?",
        height=90,
    )
    answer = st.text_area("Student answer", height=170)
    if st.button("Evaluate answer", type="primary"):
        if not answer.strip():
            st.warning("Enter an answer first.")
            return
        with st.spinner("Retrieving reference context from Endee and evaluating..."):
            result = safe_run(lambda: evaluate_answer(question, answer, filters=filters))
        if result:
            st.metric("Score", f"{result['score']}/10")
            st.markdown(result["evaluation"])
            display_sources(result["sources"])


def page_recommendations() -> None:
    st.header("Recommendations")
    progress = load_progress()
    st.metric("Saved attempts", len(progress.get("attempts", [])))
    averages = topic_averages(progress)
    if averages:
        st.dataframe(averages, use_container_width=True)
    else:
        st.info("Save quiz or evaluation scores to see weak-topic recommendations.")

    if st.button("Recommend revision topics and PYQs", type="primary"):
        with st.spinner("Finding weak topics and searching Endee PYQs..."):
            recommendations = safe_run(recommend_for_weak_topics)
        if recommendations:
            for item in recommendations:
                weak = item["weak_topic"]
                st.subheader(f"{weak['subject']} - {weak['topic']} (avg {weak['average_score']}/10)")
                for pyq in item["pyqs"]:
                    st.write(pyq.get("text", ""))
                    st.caption(source_label(pyq))
                    st.divider()
        elif recommendations is not None:
            st.success("No weak topics below 6 yet.")


def page_about() -> None:
    st.header("About Project")
    st.markdown(
        """
        **GATE Study AI Tutor** is an Endee example project that demonstrates a
        complete educational RAG system for GATE CSE preparation.

        It uses Endee as the vector database through HTTP APIs for vector insert
        and vector search, sentence-transformers for embeddings, Streamlit for
        the UI, and optional OpenAI/Gemini-compatible LLM generation with a
        fallback mode when no key is provided.
        """
    )


PAGES = {
    "Ingest Data": page_ingest,
    "AI Doubt Solver": page_doubt_solver,
    "PYQ Semantic Search": page_pyq_search,
    "Quiz Generator": page_quiz,
    "Answer Evaluator": page_evaluator,
    "Recommendations": page_recommendations,
    "About Project": page_about,
}


def main() -> None:
    st.set_page_config(page_title="GATE Study AI Tutor", layout="wide")
    st.title("GATE Study AI Tutor")
    st.caption("Advanced Educational and Exam Prep RAG System using Endee Vector Database")
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", list(PAGES))
        st.divider()
        st.caption("Start Endee, ingest data, then use retrieval pages.")
    PAGES[page]()


if __name__ == "__main__":
    main()
