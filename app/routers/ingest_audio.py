import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import transcriber

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

    segments, duration, language = transcriber.transcribe(str(audio_path))

    txt_path = transcripts_dir / f"{document_id}.txt"
    json_path = transcripts_dir / f"{document_id}.json"
    txt_path.write_text("\n".join(s["text"] for s in segments), encoding="utf-8")
    json_path.write_text(json.dumps(segments), encoding="utf-8")

    return {
        "document_id": document_id,
        "source_filename": file.filename or "",
        "num_segments": len(segments),
        "duration_seconds": duration,
        "language": language,
    }
