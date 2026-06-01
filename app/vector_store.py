from __future__ import annotations

from typing import Literal

import chromadb

from app.chunker import Chunk
from app.config import settings

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def _get_collection():
    return _get_client().get_or_create_collection(
        "documents", metadata={"hnsw:space": "cosine"}
    )


def add_chunks(
    document_id: str,
    source_filename: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    source_type: Literal["pdf", "audio"],
    extra_metadata: list[dict] | None = None,
) -> None:
    """Upsert chunks. Base metadata fields (document_id, source_filename,
    chunk_index, start_char, end_char, source_type) always override any
    extra_metadata keys with the same names."""
    if not chunks:
        return
    if extra_metadata is not None and len(extra_metadata) != len(chunks):
        raise ValueError(
            f"extra_metadata length ({len(extra_metadata)}) must match chunks length ({len(chunks)})"
        )
    collection = _get_collection()
    metadatas: list[dict] = []
    for i, c in enumerate(chunks):
        meta: dict = dict(extra_metadata[i]) if extra_metadata is not None else {}
        meta.update(
            {
                "document_id": document_id,
                "source_filename": source_filename,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "source_type": source_type,
            }
        )
        metadatas.append(meta)
    collection.upsert(
        ids=[f"{document_id}:{c.chunk_index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query(query_embedding: list[float], top_k: int) -> list[dict]:
    collection = _get_collection()
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"text": text, "metadata": meta, "distance": dist}
        for text, meta, dist in zip(documents, metadatas, distances)
    ]
