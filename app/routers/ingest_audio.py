import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import audio
from app.chunker import chunk_text
from app.embedder import embed_texts
from app.vector_store import add_chunks

router = APIRouter()

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def _data_dir() -> Path:
    return Path(os.environ.get("INGEST_DATA_DIR", "data"))


@router.post("/ingest_audio")
async def ingest_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    document_id = uuid.uuid4().hex
    data_dir = _data_dir()
    audio_dir = data_dir / "audio"
    transcripts_dir = data_dir / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"{document_id}{suffix}"
    with audio_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    segments, duration, language = audio.transcribe_segments(str(audio_path))
    segment_dicts = [asdict(s) for s in segments]

    txt_path = transcripts_dir / f"{document_id}.txt"
    json_path = transcripts_dir / f"{document_id}.json"
    txt_path.write_text("\n".join(s["text"] for s in segment_dicts), encoding="utf-8")
    json_path.write_text(json.dumps(segment_dicts), encoding="utf-8")

    JOINER = "\n\n"
    segment_offsets: list[tuple[int, int, float, float]] = []
    parts: list[str] = []
    cursor = 0
    for i, seg in enumerate(segments):
        if i > 0:
            cursor += len(JOINER)
        start = cursor
        end = cursor + len(seg.text)
        segment_offsets.append((start, end, seg.start_seconds, seg.end_seconds))
        parts.append(seg.text)
        cursor = end
    full_text = JOINER.join(parts)

    chunks = chunk_text(full_text)
    num_chunks = len(chunks)

    if chunks:
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
                {
                    "start_seconds": chunk_start_seconds,
                    "end_seconds": chunk_end_seconds,
                }
            )

        embeddings = embed_texts([c.text for c in chunks])
        add_chunks(
            document_id,
            file.filename or "",
            chunks,
            embeddings,
            source_type="audio",
            extra_metadata=extra_metadata,
        )

    return {
        "document_id": document_id,
        "source_filename": file.filename or "",
        "num_segments": len(segment_dicts),
        "num_chunks": num_chunks,
        "duration_seconds": duration,
        "language": language,
    }
