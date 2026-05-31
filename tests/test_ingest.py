import io

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app


def _make_pdf_bytes(pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_ingest_pdf_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    pdf_bytes = _make_pdf_bytes(pages=2)

    response = TestClient(app).post(
        "/ingest",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.pdf"
    assert body["page_count"] == 2
    assert "doc_id" in body
    assert "char_count" in body
    assert "num_chunks" in body
    assert isinstance(body["num_chunks"], int)

    out_file = tmp_path / f"{body['doc_id']}.txt"
    assert out_file.exists()


def test_ingest_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))

    response = TestClient(app).post(
        "/ingest",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 400
