from fastapi import FastAPI

from app.routers import ingest, query

app = FastAPI()
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}
