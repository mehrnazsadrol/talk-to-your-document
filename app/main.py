from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import tracking
from app.routers import ingest, ingest_audio, query

_STATIC_DIR = Path(__file__).parent / "static"

tracking.init()

app = FastAPI()
app.include_router(ingest.router)
app.include_router(ingest_audio.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")
