from fastapi.testclient import TestClient

import app.vector_store as vs
from app.main import app


def test_health(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr(vs, "_client", None)
    try:
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        vs._client = None
