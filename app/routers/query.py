import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app import drift, llm, tracking
from app.config import settings
from app.embedder import embed_texts
from app.vector_store import _get_collection
from app.vector_store import query as vector_query

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    # Bounds: 1..20. Below 1 short-circuits to the empty-corpus path
    # by accident; above 20 bloats the LLM prompt without quality gains.
    top_k: int | None = Field(default=None, ge=1, le=20)


@router.post("/query")
def query(req: QueryRequest):
    k = req.top_k if req.top_k is not None else settings.top_k

    with tracking.run(run_name="query", tags={"endpoint": "/query"}) as mlrun:
        mlrun.log_param("top_k", k)
        mlrun.log_param("embedding_model", settings.embedding_model)
        mlrun.log_param("llm_model", settings.llm_model)
        mlrun.log_param("corpus_size", _get_collection().count())

        mlrun.log_dict({"question": req.question, "top_k": k}, "request.json")

        t0 = time.perf_counter()
        question_embedding = embed_texts([req.question])[0]
        mlrun.log_metric("embed_ms", (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        retrieved = vector_query(question_embedding, top_k=k)
        mlrun.log_metric("retrieve_ms", (time.perf_counter() - t0) * 1000)

        top_distance = retrieved[0]["distance"] if retrieved else 1.0
        mlrun.log_metric("top_distance", top_distance)

        mlrun.log_dict(
            [
                {
                    "text": r["text"][:500],
                    "metadata": r["metadata"],
                    "distance": r["distance"],
                }
                for r in retrieved
            ],
            "retrieved.json",
        )

        if not retrieved:
            mlrun.log_metric("llm_ms", 0)
            mlrun.log_metric("prompt_tokens", 0)
            mlrun.log_metric("completion_tokens", 0)
            response_body = {"answer": "I don't know.", "sources": []}
            mlrun.log_dict(response_body, "response.json")
        else:
            t0 = time.perf_counter()
            answer, usage = llm.generate_answer(req.question, retrieved)
            mlrun.log_metric("llm_ms", (time.perf_counter() - t0) * 1000)

            if usage is None:
                mlrun.log_metric("prompt_tokens", 0)
                mlrun.log_metric("completion_tokens", 0)
                mlrun.set_tag("usage_missing", "true")
            else:
                mlrun.log_metric("prompt_tokens", usage["prompt_tokens"])
                mlrun.log_metric("completion_tokens", usage["completion_tokens"])

            source_types = sorted(
                {r["metadata"].get("source_type", "unknown") for r in retrieved}
            )
            mlrun.set_tag("source_types", ",".join(source_types))

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
            response_body = {"answer": answer, "sources": sources}
            mlrun.log_dict(response_body, "response.json")

    drift.log_query(req.question, question_embedding, top_distance)
    return response_body
