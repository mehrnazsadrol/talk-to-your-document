from __future__ import annotations

import os

import httpx
import streamlit as st

from frontend.components.citations import render_sources

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 120.0
_QUERY_TIMEOUT = 60.0


def append_turn(
    messages: list,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> list:
    """Append a chat turn to ``messages`` in place and return the list."""
    messages.append({"role": role, "content": content, "sources": sources})
    return messages


def _ingest_pdf(file) -> dict:
    resp = httpx.post(
        f"{API_BASE_URL}/ingest",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _ingest_audio(file) -> dict:
    resp = httpx.post(
        f"{API_BASE_URL}/ingest_audio",
        files={
            "file": (
                file.name,
                file.getvalue(),
                file.type or "application/octet-stream",
            )
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _query(question: str) -> dict:
    resp = httpx.post(
        f"{API_BASE_URL}/query",
        json={"question": question},
        timeout=_QUERY_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _append_source(entry: dict) -> bool:
    existing_ids = {s["document_id"] for s in st.session_state.sources}
    if entry["document_id"] in existing_ids:
        return False
    st.session_state.sources.append(entry)
    return True


st.set_page_config(page_title="Talk to Your Documents", layout="wide")

if "sources" not in st.session_state:
    st.session_state.sources = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload sources")
    pdf_files = st.file_uploader(
        "PDF", type=["pdf"], key="pdf_upload", accept_multiple_files=True
    )
    audio_files = st.file_uploader(
        "Audio",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        key="audio_upload",
        accept_multiple_files=True,
    )
    st.caption("This can take 1–2 minutes for long recordings.")

    if st.button("Ingest"):
        if not pdf_files and not audio_files:
            st.warning("Pick a PDF or audio file first.")
        else:
            for pdf_file in pdf_files or []:
                try:
                    with st.spinner("Indexing PDF..."):
                        result = _ingest_pdf(pdf_file)
                    appended = _append_source(
                        {
                            "source_filename": pdf_file.name,
                            "document_id": result["document_id"],
                            "kind": "pdf",
                            "num_chunks": result["num_chunks"],
                        }
                    )
                    if appended:
                        st.success(
                            f"Ingested PDF '{pdf_file.name}' ({result['num_chunks']} chunks)."
                        )
                    else:
                        st.info(f"'{pdf_file.name}' already ingested.")
                except httpx.HTTPStatusError as e:
                    try:
                        detail = e.response.json().get("detail")
                    except ValueError:
                        detail = None
                    st.error(f"{pdf_file.name}: {detail or e.response.text}")
                except httpx.ConnectError:
                    st.error(
                        f"Couldn't reach the backend. Is FastAPI running on {API_BASE_URL}?"
                    )
                except httpx.HTTPError as e:
                    st.error(f"{pdf_file.name}: request failed: {e}")

            for audio_file in audio_files or []:
                try:
                    with st.spinner("Transcribing audio..."):
                        result = _ingest_audio(audio_file)
                    appended = _append_source(
                        {
                            "source_filename": result.get("source_filename")
                            or audio_file.name,
                            "document_id": result["document_id"],
                            "kind": "audio",
                            "num_chunks": result["num_chunks"],
                        }
                    )
                    if appended:
                        st.success(
                            f"Ingested audio '{audio_file.name}' ({result['num_chunks']} chunks)."
                        )
                    else:
                        st.info(f"'{audio_file.name}' already ingested.")
                except httpx.HTTPStatusError as e:
                    try:
                        detail = e.response.json().get("detail")
                    except ValueError:
                        detail = None
                    st.error(f"{audio_file.name}: {detail or e.response.text}")
                except httpx.ConnectError:
                    st.error(
                        f"Couldn't reach the backend. Is FastAPI running on {API_BASE_URL}?"
                    )
                except httpx.HTTPError as e:
                    st.error(f"{audio_file.name}: request failed: {e}")

    st.divider()

    if st.session_state.sources:
        st.subheader("Ingested this session")
        for s in st.session_state.sources:
            st.write(f"{s['source_filename']} · {s['kind']} · {s['num_chunks']} chunks")

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Local RAG · DeepSeek · ChromaDB")

st.title("Talk to Your Documents")
st.caption("Ask questions across your PDFs and audio. Answers cite the exact source.")

if not st.session_state.sources:
    with st.container():
        st.write("Upload a PDF or audio file from the sidebar to get started.")
        st.write("Try: a research paper, a podcast episode, a lecture recording.")

if st.session_state.sources and not st.session_state.messages:
    suggestions = [
        "What is this document about?",
        "Summarise the key claims.",
        "What evidence is given for X?",
    ]
    cols = st.columns(3)
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, key=f"suggest_{suggestion}"):
            st.session_state._suggested = suggestion
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources") or [])

prompt = st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.sources,
)
if not prompt and st.session_state.get("_suggested"):
    prompt = st.session_state._suggested
    del st.session_state._suggested
if prompt:
    append_turn(st.session_state.messages, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your sources..."):
                resp = _query(prompt)
            st.markdown(resp["answer"])
            render_sources(resp["sources"])
            append_turn(
                st.session_state.messages,
                "assistant",
                resp["answer"],
                sources=resp["sources"],
            )
        except httpx.HTTPStatusError as e:
            st.error(f"Query failed: {e.response.text}")
        except httpx.ConnectError:
            st.error(
                f"Couldn't reach the backend. Is FastAPI running on {API_BASE_URL}?"
            )
        except httpx.HTTPError as e:
            st.error(f"Request failed: {e}")
