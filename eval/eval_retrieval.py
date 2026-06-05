from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from pypdf import PdfReader

from app import audio, tracking, vector_store
from app.chunker import chunk_text
from app.config import settings
from app.embedder import embed_texts

PDF_SUFFIXES = {".pdf"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def load_golden_set(path: Path) -> list[dict]:
    """Read a JSONL golden set; one record per line."""
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _doc_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _seed_pdf(path: Path) -> None:
    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(text)
    if not chunks:
        return
    embeddings = embed_texts([c.text for c in chunks])
    vector_store.add_chunks(
        _doc_id(path), path.name, chunks, embeddings, source_type="pdf"
    )


def _seed_audio(path: Path) -> None:
    segments, _duration, _language = audio.transcribe_segments(str(path))
    if not segments:
        return

    JOINER = "\n\n"
    segment_offsets: list[tuple[int, int, float, float]] = []
    parts: list[str] = []
    cursor = 0
    for i, seg in enumerate(segments):
        start = cursor
        if i < len(segments) - 1:
            end = cursor + len(seg.text) + len(JOINER)
        else:
            end = cursor + len(seg.text)
        segment_offsets.append((start, end, seg.start_seconds, seg.end_seconds))
        parts.append(seg.text)
        cursor = end
    full_text = JOINER.join(parts)

    chunks = chunk_text(full_text)
    if not chunks:
        return

    extra_metadata: list[dict] = []
    for c in chunks:
        overlapping = [
            (s_start, s_end, t_start, t_end)
            for (s_start, s_end, t_start, t_end) in segment_offsets
            if s_start < c.end_char and s_end > c.start_char
        ]
        if overlapping:
            chunk_start_seconds = overlapping[0][2]
            chunk_end_seconds = overlapping[-1][3]
        else:
            preceding = [
                (s_start, s_end, t_start, t_end)
                for (s_start, s_end, t_start, t_end) in segment_offsets
                if s_end <= c.start_char
            ]
            fallback = preceding[-1][3] if preceding else 0.0
            chunk_start_seconds = fallback
            chunk_end_seconds = fallback
        extra_metadata.append(
            {"start_seconds": chunk_start_seconds, "end_seconds": chunk_end_seconds}
        )

    embeddings = embed_texts([c.text for c in chunks])
    vector_store.add_chunks(
        _doc_id(path),
        path.name,
        chunks,
        embeddings,
        source_type="audio",
        extra_metadata=extra_metadata,
    )


def seed_corpus(persist_dir: Path) -> None:
    settings.chroma_persist_dir = str(persist_dir)
    vector_store._client = None

    source_dir = Path("data/golden")
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            _seed_pdf(path)
        elif suffix in AUDIO_SUFFIXES:
            _seed_audio(path)


def hit_rate_at_k(
    retrieved: list[dict],
    expected_source_filename: str,
    expected_chunk_indices: list[int],
    k: int,
) -> bool:
    """True if any of the top-k retrieved chunks matches the expected filename
    AND has a chunk_index in the expected list."""
    expected = set(expected_chunk_indices)
    for hit in retrieved[:k]:
        meta = hit["metadata"]
        if (
            meta.get("source_filename") == expected_source_filename
            and meta.get("chunk_index") in expected
        ):
            return True
    return False


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) < 20:
        return max(values)
    return statistics.quantiles(values, n=20)[18]


def run_eval(golden_path: Path, results_dir: Path, top_k_max: int = 5) -> dict:
    questions = load_golden_set(golden_path)

    results_dir.mkdir(parents=True, exist_ok=True)

    per_question: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir)
        seed_corpus(persist_dir)

        for q in questions:
            query_embedding = embed_texts([q["question"]])[0]
            t0 = time.perf_counter()
            retrieved = vector_store.query(query_embedding, top_k=top_k_max)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            hit3 = hit_rate_at_k(
                retrieved, q["expected_source_filename"], q["expected_chunk_indices"], 3
            )
            hit5 = hit_rate_at_k(
                retrieved, q["expected_source_filename"], q["expected_chunk_indices"], 5
            )
            top5 = [
                {
                    "source_filename": h["metadata"].get("source_filename"),
                    "chunk_index": h["metadata"].get("chunk_index"),
                    "distance": h["distance"],
                }
                for h in retrieved[:5]
            ]
            per_question.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "expected_source_filename": q["expected_source_filename"],
                    "expected_chunk_indices": q["expected_chunk_indices"],
                    "hit@3": hit3,
                    "hit@5": hit5,
                    "top5": top5,
                    "latency_ms": latency_ms,
                }
            )

    latencies = [pq["latency_ms"] for pq in per_question]
    n = len(per_question)
    aggregate = {
        "hit_rate_at_3": (
            sum(1 for pq in per_question if pq["hit@3"]) / n if n else 0.0
        ),
        "hit_rate_at_5": (
            sum(1 for pq in per_question if pq["hit@5"]) / n if n else 0.0
        ),
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": _p95(latencies),
    }

    payload = {
        "aggregate": aggregate,
        "per_question": per_question,
        "params": {
            "embedding_model": settings.embedding_model,
            "top_k_max": top_k_max,
            "golden_set_path": str(golden_path),
            "golden_set_size": n,
        },
    }

    timestamp = (
        datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "-")
    )
    out_path = results_dir / f"{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with tracking.run("eval_retrieval", tags={"golden_set_size": str(n)}) as r:
        r.log_param("embedding_model", settings.embedding_model)
        r.log_param("top_k_max", top_k_max)
        r.log_param("golden_set_path", str(golden_path))
        r.log_metric("hit_rate_at_3", aggregate["hit_rate_at_3"])
        r.log_metric("hit_rate_at_5", aggregate["hit_rate_at_5"])
        r.log_metric("p50_latency_ms", aggregate["p50_latency_ms"])
        r.log_metric("p95_latency_ms", aggregate["p95_latency_ms"])
        r.log_dict(payload, "results.json")

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set", type=Path, default=Path("eval/golden_set.jsonl")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    tracking.init()
    aggregate = run_eval(args.golden_set, args.results_dir)
    print(json.dumps(aggregate))


if __name__ == "__main__":
    main()
