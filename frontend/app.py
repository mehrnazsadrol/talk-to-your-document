"""Streamlit frontend for Talk to Your Documents.

Run the backend and frontend in two terminals:

    uvicorn app.main:app --reload
    streamlit run frontend/app.py

The frontend talks to the backend at ``API_BASE_URL`` (default
``http://localhost:8000``). Override for a non-default port::

    API_BASE_URL=http://localhost:9000 streamlit run frontend/app.py

This ticket (2.5) scaffolds layout, uploaders, and HTTP plumbing only.
Chat loop, citations, and polish are wired up in 2.6 / 2.7 / 2.8.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 120.0


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
        files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _append_source(entry: dict) -> None:
    existing_ids = {s["document_id"] for s in st.session_state.sources}
    if entry["document_id"] not in existing_ids:
        st.session_state.sources.append(entry)


st.set_page_config(page_title="Talk to Your Documents", layout="wide")

if "sources" not in st.session_state:
    st.session_state.sources = []

with st.sidebar:
    st.header("Upload sources")
    pdf_file = st.file_uploader("PDF", type=["pdf"], key="pdf_upload")
    audio_file = st.file_uploader(
        "Audio",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="audio_upload",
    )

    if st.button("Ingest"):
        if pdf_file is None and audio_file is None:
            st.warning("Pick a PDF or audio file first.")
        else:
            if pdf_file is not None:
                try:
                    result = _ingest_pdf(pdf_file)
                    _append_source(
                        {
                            "source_filename": pdf_file.name,
                            "document_id": result["document_id"],
                            "kind": "pdf",
                            "num_chunks": result["num_chunks"],
                        }
                    )
                    st.success(
                        f"Ingested PDF '{pdf_file.name}' ({result['num_chunks']} chunks)."
                    )
                except httpx.HTTPStatusError as e:
                    st.error(e.response.text)
                except httpx.HTTPError as e:
                    st.error(f"Request failed: {e}")

            if audio_file is not None:
                try:
                    result = _ingest_audio(audio_file)
                    _append_source(
                        {
                            "source_filename": result.get("source_filename") or audio_file.name,
                            "document_id": result["document_id"],
                            "kind": "audio",
                            "num_chunks": result["num_chunks"],
                        }
                    )
                    st.success(
                        f"Ingested audio '{audio_file.name}' ({result['num_chunks']} chunks)."
                    )
                except httpx.HTTPStatusError as e:
                    st.error(e.response.text)
                except httpx.HTTPError as e:
                    st.error(f"Request failed: {e}")

    if st.session_state.sources:
        st.subheader("Ingested this session")
        for s in st.session_state.sources:
            st.write(f"- [{s['kind']}] {s['source_filename']} ({s['num_chunks']} chunks)")

st.title("Talk to Your Documents")
st.caption("Ask questions across the PDFs and audio you ingest.")

if not st.session_state.sources:
    st.info("Ingest a document from the sidebar to get started.")
