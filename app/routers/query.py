from fastapi import APIRouter
from pydantic import BaseModel

from app import llm
from app.config import settings
from app.embedder import embed_texts
from app.vector_store import query as vector_query

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


@router.post("/query")
def query(req: QueryRequest):
    k = req.top_k if req.top_k is not None else settings.top_k
    question_embedding = embed_texts([req.question])[0]
    retrieved = vector_query(question_embedding, top_k=k)

    if not retrieved:
        return {"answer": "I don't know.", "sources": []}

    answer = llm.generate_answer(req.question, retrieved)

    sources = [
        {
            "document_id": r["metadata"]["document_id"],
            "chunk_index": r["metadata"]["chunk_index"],
            "source_filename": r["metadata"]["source_filename"],
            "snippet": r["text"][:200],
            "distance": r["distance"],
        }
        for r in retrieved
    ]
    return {"answer": answer, "sources": sources}
