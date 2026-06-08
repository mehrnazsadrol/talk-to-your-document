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

A RAG demo that answers questions over your PDFs and audio with cited sources.
DeepSeek for generation, sentence-transformers for embeddings, ChromaDB for retrieval,
faster-whisper for audio transcription. Streamlit UI, FastAPI backend.

Source: https://github.com/mehrnazsadr/talk-to-your-document

**Storage note:** the free HF Spaces tier wipes the container's filesystem on every
restart, so the ChromaDB corpus is ephemeral — you'll need to re-ingest your PDFs
and audio after each cold start.