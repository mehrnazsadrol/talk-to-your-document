from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# Ingest can include Whisper transcription + bulk embedding, both multi-second.
_TIMEOUT = 120.0
# Single LLM round-trip; 60s is generous for DeepSeek.
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
    """Append ``entry`` to the sidebar source list unless its document_id is
    already present. Returns True if appended, False if it was a duplicate."""
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

    if st.button("Ingest"):
        if not pdf_files and not audio_files:
            st.warning("Pick a PDF or audio file first.")
        else:
            for pdf_file in pdf_files or []:
                try:
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
                    st.error(f"{pdf_file.name}: {e.response.text}")
                except httpx.HTTPError as e:
                    st.error(f"{pdf_file.name}: request failed: {e}")

            for audio_file in audio_files or []:
                try:
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
                    st.error(f"{audio_file.name}: {e.response.text}")
                except httpx.HTTPError as e:
                    st.error(f"{audio_file.name}: request failed: {e}")

    if st.session_state.sources:
        st.subheader("Ingested this session")
        for s in st.session_state.sources:
            st.write(
                f"- [{s['kind']}] {s['source_filename']} ({s['num_chunks']} chunks)"
            )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Talk to Your Documents")
st.caption("Ask questions across the PDFs and audio you ingest.")

if not st.session_state.sources and not st.session_state.messages:
    st.info("Ingest a document from the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    # Persist user turn before the LLM call so a failure doesn't lose the question.
    append_turn(st.session_state.messages, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                resp = _query(prompt)
            st.markdown(resp["answer"])
            append_turn(
                st.session_state.messages,
                "assistant",
                resp["answer"],
                sources=resp["sources"],
            )
        except httpx.HTTPStatusError as e:
            st.error(f"Query failed: {e.response.text}")
        except httpx.HTTPError as e:
            st.error(f"Request failed: {e}")
