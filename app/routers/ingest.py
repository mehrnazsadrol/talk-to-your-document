import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pypdf import PdfReader

from app.chunker import chunk_text
from app.embedder import embed_texts
from app.text_cleaning import strip_front_matter
from app.vector_store import add_chunks

router = APIRouter()


def _data_dir() -> Path:
    path = Path(os.environ.get("INGEST_DATA_DIR", "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.post("/ingest")
def ingest(file: UploadFile = File(...)):
    if file.content_type != "application/pdf" or not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    reader = PdfReader(file.file)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    text = strip_front_matter(text)

    document_id = uuid.uuid4().hex
    out_path = _data_dir() / f"{document_id}.txt"
    out_path.write_text(text, encoding="utf-8")

    chunks = chunk_text(text)
    embeddings = embed_texts([c.text for c in chunks])
    add_chunks(document_id, file.filename or "", chunks, embeddings, source_type="pdf")

    return {
        "document_id": document_id,
        "num_pages": len(reader.pages),
        "num_chunks": len(chunks),
    }
