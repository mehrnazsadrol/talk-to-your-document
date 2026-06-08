---
title: Talk to Your Documents
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
app_file: Dockerfile.hfspace
---

# Talk to Your Documents

Ask questions across PDFs and audio. Cited answers. Multimodal RAG with FastAPI + ChromaDB + DeepSeek + Whisper.

[![CI](https://github.com/mehrnazsadrol/talk-to-your-document/actions/workflows/ci.yml/badge.svg)](https://github.com/mehrnazsadrol/talk-to-your-document/actions/workflows/ci.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Demo](https://img.shields.io/badge/demo-HuggingFace-blue)](https://huggingface.co/spaces/MehrnazSadroleslami/talk-to-your-document)

---

## Demo

- **Live demo:** https://huggingface.co/spaces/MehrnazSadroleslami/talk-to-your-document
- **Walkthrough (≤2 min):** _Loom link coming — ticket 4.6._
- **At a glance:**

![Demo](docs/demo.gif)

> Free-tier HF Space; cold-start ~30s; the corpus resets on restart, so re-ingest your files after each cold boot.

---

## How it works

A user uploads a PDF or audio file. The backend extracts text (pypdf for PDFs, faster-whisper for audio), splits it into ~500-token chunks with 50-token overlap, embeds each chunk with `all-MiniLM-L6-v2` (384-dim), and writes the vectors plus metadata to a persistent ChromaDB collection. On a question, the same embedder vectorises the query, ChromaDB returns the top-k nearest chunks by cosine distance, and DeepSeek's chat-completion API generates a grounded answer with inline `[chunk_index=N]` citations. The frontend renders the answer plus a collapsed "Sources" expander showing filename, locator (`mm:ss` for audio), snippet, and distance.

![Architecture](docs/architecture.png)

---

## Key numbers

| Metric | Value | Notes |
|---|---|---|
| Retrieval hit-rate@5 | _TODO_ | 15-question golden set; run `python -m eval.eval_retrieval` |
| Retrieval hit-rate@3 | _TODO_ | same set |
| p95 query latency | _TODO_ ms | embed + retrieve + generate; from `eval/results/*.json` |
| Test coverage | 81% | `pytest --cov=app --cov=eval` |
| Tests passing | 69 / 69 | `pytest -q` |

_Update these after each meaningful change — stale numbers are worse than no numbers._

---

## Run it locally

### With Docker Compose (recommended)

```bash
cp .env.example .env
# paste your DeepSeek API key into .env
docker compose up --build
# open http://localhost:8501
```

Two containers: `api` (FastAPI on :8000) + `frontend` (Streamlit on :8501). Chroma persists in a named volume; `docker compose down -v` wipes it.

### From source

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then paste your DEEPSEEK_API_KEY

# terminal 1
uvicorn app.main:app --reload

# terminal 2
streamlit run frontend/app.py
```

Streamlit on http://localhost:8501; FastAPI on http://localhost:8000 (also serves a minimal static dev UI at `/`).

### Run the eval

```bash
source .venv/bin/activate
python -m eval.eval_retrieval
# results: eval/results/<utc>.json
# mlflow: mlflow ui --backend-store-uri ./mlruns
```

Drop your own PDF + audio fixtures into `data/golden/` first; the schema for `eval/golden_set.jsonl` is `{id, question, expected_source_filename, expected_chunk_indices, notes}`.

### Run the tests

```bash
pytest -q -m "not slow"   # fast lane (default for CI)
pytest -q -m slow         # slow lane (loads the real ~80MB embedder)
```

---

## Key decisions

A 5-bullet preview; full reasoning + experiments lives in `docs/decisions.md`.

- **`all-MiniLM-L6-v2` for embeddings** — small (~80MB), fast, 384-dim, good enough for portfolio-scale corpora. Hit-rate degrades on documents with heavy shared vocabulary; mpnet-base-v2 or a cross-encoder reranker is the documented next step.
- **ChromaDB local persistent client** — zero-ops, SQLite-backed, survives restarts. Would swap for pgvector at multi-tenant scale.
- **Character-recursive chunking (500 tokens / 50 overlap)** — beat semantic chunking on the golden set; offsets are exact so `chunk.text == original[start_char:end_char]` always holds.
- **DeepSeek for generation** — OpenAI-compatible API, ~10× cheaper than GPT-4-class models, no vendor lock-in. Swappable via `settings.llm_model`.
- **`faster-whisper` for audio** — CTranslate2 backend, ~4× faster than reference Whisper on CPU. CPU-only `int8` quantisation for the free HF tier.

---

## What I'd do next

Honest list of known gaps and future work — the part interviewers ask about.

- **Capture `page_number` per PDF chunk** (ticket 1.4 follow-up). Today PDF citations are filename-only; audio citations already show `mm:ss`.
- **Idempotent `document_id`** via `sha256(file_bytes)[:16]` (code-review item #43). Today re-uploading the same PDF creates duplicate chunks, eating retrieval top-k budget.
- **Embedder upgrade** to `all-mpnet-base-v2` (768-dim) or add a cross-encoder reranker (`bge-reranker-base`) on top of vector search. Measure hit-rate@1 lift on the existing golden set.
- **Persistent HF Space storage** so the demo corpus survives cold restarts. Free-tier wipes `/data` on every restart.
- **Streaming the LLM response** through Server-Sent Events so the UI feels live for long answers.
- **Clickable citations** that scroll the source PDF / audio to the cited offset in a side panel.

---

## Repo layout

```
talk-to-your-document/
├── app/                     FastAPI backend
│   ├── main.py              app factory; GET /, /health
│   ├── config.py            pydantic-settings (env-driven)
│   ├── chunker.py           recursive character chunker
│   ├── embedder.py          sentence-transformers wrapper
│   ├── transcriber.py       faster-whisper wrapper
│   ├── audio.py             silence-based splitter + transcribe_segments
│   ├── vector_store.py      ChromaDB add_chunks / query
│   ├── llm.py               DeepSeek (OpenAI-compatible) wrapper
│   ├── tracking.py          MLflow init + run() context
│   ├── drift.py             query-embedding drift detection
│   ├── static/index.html    minimal dev UI at GET /
│   └── routers/             POST /ingest, /ingest_audio, /query
├── frontend/                Streamlit recruiter-facing UI
│   ├── app.py               chat loop, sidebar uploaders
│   └── components/          citation renderers
├── eval/                    Golden-set retrieval eval
│   ├── eval_retrieval.py    seed_corpus + hit_rate_at_k + run_eval
│   ├── golden_set.jsonl     15 Q&A pairs across 4 PDFs
│   └── results/             timestamped per-run JSON
├── scripts/check_drift.py   weekly drift CLI
├── tests/                   pytest suite (69 tests, 81% coverage)
├── hfspace/start.sh         HF Space process manager (uvicorn + streamlit)
├── Dockerfile               HF Space single-container build
├── Dockerfile.api           local-dev API image (used by compose)
├── frontend/Dockerfile      local-dev Streamlit image
├── docker-compose.yml       api + frontend services with shared volume
├── .github/workflows/ci.yml lint + test + docker build
├── ruff.toml                lint + format config
├── pytest.ini               coverage gate (80%) + markers
└── requirements.txt         all Python deps
```

---

## License

MIT — see [LICENSE](LICENSE).
