"""PharmaLens Chat UI — Streamlit frontend for the RAG backend.

Connects to the FastAPI backend (src/api.py) and provides:
  - Question input with submit on Enter or button click
  - Streaming answer that appears token-by-token
  - Citations displayed before generation starts
  - Source cards with document name, chunk ID, relevance score, and source text
  - Status badge (answered / no_context / insufficient_relevance)
  - Graceful handling of stream interruptions with partial text preserved
  - Document upload that indexes new content at runtime
  - Loading states and clear error messages throughout

Run with:
    streamlit run src/ui.py

Set RAG_API_URL in the environment or .env to point at a non-local backend.
"""

from __future__ import annotations

import json
import os

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

def stream_question(question: str):
    """
    POST /query/stream and yield parsed SSE event dicts.

    Yields dicts with keys: type, and then text / sources / value / message
    depending on the event type.
    """
    with requests.post(
        f"{API_URL}/query/stream",
        json={"question": question},
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                continue


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

STATUS_CONFIG: dict[str, tuple[str, str]] = {
    "answered":               ("Answered",          "normal"),
    "no_context":             ("No context found",  "warning"),
    "insufficient_relevance": ("Low relevance",     "warning"),
    "unknown":                ("Unknown status",    "warning"),
}


def render_status_badge(status: str) -> None:
    label, kind = STATUS_CONFIG.get(status, ("Unknown status", "warning"))
    if kind == "normal":
        st.success(f"✅ {label}")
    else:
        st.warning(f"⚠️ {label}")


def render_sources(sources: list[dict]) -> None:
    """Render citation cards. Each card shows doc name, chunk ID, score, and source text."""
    if not sources:
        st.caption("No sources were cited for this answer.")
        return

    st.markdown("**Sources**")
    for src in sources:
        label      = src.get("label", "")
        document   = src.get("document") or "Unknown source"
        chunk_id   = src.get("chunk_id")
        score      = src.get("score")
        text       = src.get("text", "")

        header = f"{label} {document}" if label else document
        with st.expander(header, expanded=False):
            if chunk_id:
                st.markdown(f"**Chunk ID:** `{chunk_id}`")
            if score is not None:
                bar_value = min(max(float(score), 0.0), 1.0)
                st.markdown(f"**Relevance score:** {score:.2f}")
                st.progress(bar_value)
            else:
                st.caption("Score not available.")
            if text:
                st.divider()
                st.caption("Source text:")
                st.markdown(f"> {text[:600]}{'...' if len(text) > 600 else ''}")


def render_completed_entry(entry: dict) -> None:
    """Render a finished history entry (non-streaming)."""
    status  = entry.get("status", "unknown")
    answer  = entry.get("answer", "")
    sources = entry.get("sources", [])

    render_status_badge(status)
    st.markdown(answer)
    st.divider()
    render_sources(sources)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PharmaLens",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💊 PharmaLens")
    st.caption("Pharmaceutical Research Intelligence")
    st.divider()

    if check_health():
        st.success("Backend connected", icon="🟢")
    else:
        st.error(f"Backend unreachable\n{API_URL}", icon="🔴")

    st.divider()
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
            with st.status(f"Indexing {uploaded_file.name}...", expanded=True) as upload_status:
                try:
                    result = upload_document(file_bytes, uploaded_file.name)
                    summary = result.get("summary", {})
                    st.write(f"**{result.get('filename')}** indexed successfully.")
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

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    # Each entry: {question, answer, sources, status, complete, error}
    st.session_state.history: list[dict] = []

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("PharmaLens Research Assistant")
st.caption("Ask a question about your pharmaceutical research corpus.")

with st.form(key="query_form", clear_on_submit=True):
    question_input = st.text_input(
        label="Your question",
        placeholder="e.g. What did Study 001 evaluate?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", use_container_width=True)

if submitted:
    q = question_input.strip()
    if not q:
        st.warning("Please enter a question.")
    elif len(q) < 3:
        st.warning("Question must be at least 3 characters.")
    else:
        # Push a placeholder entry to the top of history
        entry: dict = {
            "question": q,
            "answer": "",
            "sources": [],
            "status": "unknown",
            "complete": False,
            "error": None,
        }
        st.session_state.history.insert(0, entry)

        # Stream into the top entry
        with st.container(border=True):
            st.markdown(f"**Q: {q}**")

            sources_placeholder = st.empty()
            status_placeholder  = st.empty()
            answer_placeholder  = st.empty()
            error_placeholder   = st.empty()

            accumulated_answer = ""
            stream_error       = None
            stream_complete    = False

            try:
                for event in stream_question(q):
                    etype = event.get("type")

                    if etype == "citations":
                        entry["sources"] = event.get("sources", [])
                        with sources_placeholder.container():
                            render_sources(entry["sources"])

                    elif etype == "status":
                        entry["status"] = event.get("value", "unknown")
                        with status_placeholder.container():
                            render_status_badge(entry["status"])

                    elif etype == "token":
                        accumulated_answer += event.get("text", "")
                        entry["answer"] = accumulated_answer
                        answer_placeholder.markdown(accumulated_answer + " ▌")

                    elif etype == "error":
                        stream_error = event.get("message", "An error occurred.")
                        error_placeholder.error(
                            f"{stream_error}  \n\n"
                            "The answer above may be incomplete. "
                            "You can re-submit the question to retry."
                        )

                    elif etype == "done":
                        stream_complete = True

                # Remove the typing cursor once done
                answer_placeholder.markdown(accumulated_answer)

            except requests.exceptions.ConnectionError:
                stream_error = (
                    f"Could not connect to the backend at **{API_URL}**. "
                    "Is the server running?"
                )
                error_placeholder.error(stream_error)

            except requests.exceptions.Timeout:
                stream_error = "The request timed out. The backend may be under load — try again."
                error_placeholder.error(stream_error)

            except requests.exceptions.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:
                    pass
                stream_error = f"Backend error: {detail or str(exc)}"
                error_placeholder.error(stream_error)

            except requests.exceptions.RequestException as exc:
                stream_error = f"Request failed: {exc}"
                error_placeholder.error(stream_error)

            finally:
                entry["complete"] = stream_complete
                entry["error"]    = stream_error

# ---------------------------------------------------------------------------
# Conversation history (entries after the first / already completed)
# ---------------------------------------------------------------------------

history_to_show = (
    st.session_state.history[1:]
    if st.session_state.history and not st.session_state.history[0].get("complete", True) is False
    else st.session_state.history
)

# Show all completed history entries
for entry in st.session_state.history:
    if not entry.get("complete"):
        continue
    with st.container(border=True):
        st.markdown(f"**Q: {entry['question']}**")
        render_completed_entry(entry)

if not st.session_state.history:
    st.info("Ask a question above to get started.", icon="💬")
elif any(e.get("complete") for e in st.session_state.history):
    if st.button("Clear history", use_container_width=False):
        st.session_state.history = []
        st.rerun()
