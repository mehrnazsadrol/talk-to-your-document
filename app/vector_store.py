"""Persistent ChromaDB-backed vector store.

The module-level `_client` is cached lazily by `_get_client()`. Tests that
need to point at a different `settings.chroma_persist_dir` (e.g. a
`tmp_path`) can reset the cache by setting `vector_store._client = None`
after monkeypatching the setting, which forces the next call to
re-construct the client against the new path. This is what simulates a
"server restart" against the same on-disk database.
"""

from __future__ import annotations

import chromadb

from app.chunker import Chunk
from app.config import settings

_client: chromadb.api.client.Client | None = None


def _get_client() -> chromadb.api.client.Client:
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
) -> None:
    if not chunks:
        return
    collection = _get_collection()
    collection.upsert(
        ids=[f"{document_id}:{c.chunk_index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "source_filename": source_filename,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ],
    )


def query(query_embedding: list[float], top_k: int) -> list[dict]:
    collection = _get_collection()
    result = collection.query(
        query_embeddings=[query_embedding], n_results=top_k
    )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"text": text, "metadata": meta, "distance": dist}
        for text, meta, dist in zip(documents, metadatas, distances)
    ]
