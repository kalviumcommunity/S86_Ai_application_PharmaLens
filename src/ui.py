"""PharmaLens Chat UI — Streamlit frontend for the RAG backend.

Connects to the FastAPI backend (src/api.py) and provides:
  - Question input with submit on Enter or button click
  - Grounded answer display with citation markers
  - Source cards showing document name, chunk ID, and relevance score
  - Status badge (answered / no_context / insufficient_relevance)
  - Document upload that indexes new content at runtime
  - Loading spinners and clear error messages

Run with:
    streamlit run src/ui.py

The backend URL defaults to http://localhost:8000 and can be overridden
by setting RAG_API_URL in the environment or a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
SUPPORTED_UPLOAD_TYPES = ["txt", "md", "html"]
MAX_UPLOAD_MB = 10

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def ask_question(question: str) -> dict:
    """POST /query and return the parsed JSON response."""
    response = requests.post(
        f"{API_URL}/query",
        json={"question": question},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """POST /documents with a multipart file upload."""
    response = requests.post(
        f"{API_URL}/documents",
        files={"file": (filename, file_bytes)},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def check_health() -> bool:
    """Return True if the backend is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

STATUS_CONFIG = {
    "answered": ("✅ Answered", "normal"),
    "no_context": ("⚠️ No context found", "inverse"),
    "insufficient_relevance": ("⚠️ Low relevance", "inverse"),
    "unknown": ("❓ Unknown status", "inverse"),
}


def render_status_badge(status: str) -> None:
    label, badge_type = STATUS_CONFIG.get(status, ("❓ Unknown", "inverse"))
    if badge_type == "normal":
        st.success(label)
    else:
        st.warning(label)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No sources were cited for this answer.")
        return

    st.markdown("**Sources**")
    for i, src in enumerate(sources, start=1):
        source_name = src.get("source") or "Unknown source"
        chunk_id = src.get("chunk_id")
        score = src.get("score")

        with st.expander(f"[{i}] {source_name}", expanded=False):
            if chunk_id:
                st.markdown(f"**Chunk ID:** `{chunk_id}`")
            if score is not None:
                bar_value = min(max(float(score), 0.0), 1.0)
                st.markdown(f"**Relevance score:** {score:.2f}")
                st.progress(bar_value)
            else:
                st.caption("Score not available.")


def render_answer(result: dict) -> None:
    status = result.get("status", "unknown")
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    render_status_badge(status)
    st.markdown("### Answer")
    st.markdown(answer)
    st.divider()
    render_sources(sources)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PharmaLens",
    page_icon="💊",
    layout="wide",
)

# -- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.title("💊 PharmaLens")
    st.caption("Pharmaceutical Research Intelligence")
    st.divider()

    # Backend health
    if check_health():
        st.success("Backend connected", icon="🟢")
    else:
        st.error(f"Backend unreachable\n{API_URL}", icon="🔴")

    st.divider()

    # Document upload
    st.subheader("📄 Upload Document")
    st.caption(f"Supported: {', '.join(SUPPORTED_UPLOAD_TYPES)} · Max {MAX_UPLOAD_MB} MB")

    uploaded_file = st.file_uploader(
        label="Choose a file",
        type=SUPPORTED_UPLOAD_TYPES,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("Index document", use_container_width=True):
            file_bytes = uploaded_file.read()
            with st.status(f"Indexing {uploaded_file.name}…", expanded=True) as upload_status:
                try:
                    result = upload_document(file_bytes, uploaded_file.name)
                    summary = result.get("summary", {})
                    st.write(f"✅ **{result.get('filename')}** indexed successfully.")
                    st.write(f"Chunks produced: **{summary.get('chunks_produced')}**")
                    st.write(f"Chunks indexed:  **{summary.get('chunks_indexed')}**")
                    upload_status.update(label="Indexed", state="complete")
                except requests.exceptions.HTTPError as exc:
                    detail = ""
                    try:
                        detail = exc.response.json().get("detail", "")
                    except Exception:
                        pass
                    upload_status.update(label="Upload failed", state="error")
                    st.error(f"Upload failed: {detail or str(exc)}")
                except requests.exceptions.RequestException as exc:
                    upload_status.update(label="Upload failed", state="error")
                    st.error(f"Could not reach the backend: {exc}")

    st.divider()
    st.caption(f"API: `{API_URL}`")

# -- Main area ---------------------------------------------------------------
st.title("PharmaLens Research Assistant")
st.caption("Ask a question about your pharmaceutical research corpus.")

# Initialise session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "result": dict}

# Question form — submits on Enter or button click
with st.form(key="query_form", clear_on_submit=True):
    question = st.text_input(
        label="Your question",
        placeholder="e.g. What did Study 001 evaluate?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", use_container_width=True)

if submitted:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
    elif len(question) < 3:
        st.warning("Question must be at least 3 characters.")
    else:
        with st.spinner("Searching the corpus and generating an answer…"):
            try:
                result = ask_question(question)
                st.session_state.history.insert(0, {"question": question, "result": result})
            except requests.exceptions.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:
                    pass
                st.error(f"The backend returned an error: {detail or str(exc)}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to the backend at **{API_URL}**. "
                    "Is the server running?"
                )
            except requests.exceptions.Timeout:
                st.error("The request timed out. The backend may be under load — try again.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

# -- Conversation history ----------------------------------------------------
if st.session_state.history:
    for entry in st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**Q: {entry['question']}**")
            render_answer(entry["result"])

    if st.button("Clear history", use_container_width=False):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Ask a question above to get started.", icon="💬")
