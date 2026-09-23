from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

import app.main as main


class FakeService:
    def __init__(self, embeddings=None, error=None):
        self.embeddings = embeddings or [[0.1, 0.2]]
        self.error = error

    def embed(self, texts):
        if self.error:
            raise self.error
        return self.embeddings


class FakeStore:
    def __init__(self, error=None):
        self.error = error
        self.received = None

    def add_chunks(self, document_id, filename, chunks, embeddings):
        if self.error:
            raise self.error
        self.received = (document_id, filename, chunks, embeddings)


def create_text_pdf(path: Path):
    content = b"BT /F1 12 Tf 72 720 Td (API upload test document) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(pdf)


def create_empty_pdf(path: Path):
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with path.open("wb") as handle:
        writer.write(handle)


def client_for(monkeypatch, tmp_path, service=None, store=None):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    main.UPLOAD_DIR.mkdir()
    monkeypatch.setattr(main, "ai_service", lambda: service or FakeService())
    monkeypatch.setattr(main, "store", store or FakeStore())
    return TestClient(main.app)


def test_valid_pdf_upload_returns_ingestion_details(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    create_text_pdf(pdf_path)
    store = FakeStore()

    with client_for(monkeypatch, tmp_path, store=store) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("document.pdf", pdf_path.read_bytes(), "application/pdf")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["document_id"]
    assert body["pages_processed"] == 1
    assert body["chunks_created"] == 1
    assert store.received is not None


def test_invalid_file_extension_returns_400(monkeypatch, tmp_path):
    with client_for(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("notes.txt", b"not a PDF", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_oversized_upload_returns_413(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MAX_FILE_SIZE", 10)

    with client_for(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("large.pdf", b"x" * 11, "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "File exceeds the 10 MB limit."


def test_empty_pdf_returns_controlled_error(monkeypatch, tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    create_empty_pdf(pdf_path)

    with client_for(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("empty.pdf", pdf_path.read_bytes(), "application/pdf")},
        )

    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]


def test_gemini_failure_returns_controlled_error(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    create_text_pdf(pdf_path)
    service = FakeService(error=RuntimeError("embedding service unavailable"))

    with client_for(monkeypatch, tmp_path, service=service) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("document.pdf", pdf_path.read_bytes(), "application/pdf")},
        )

    assert response.status_code == 422
    assert "embedding service unavailable" in response.json()["detail"]


def test_storage_failure_returns_controlled_error(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    create_text_pdf(pdf_path)
    store = FakeStore(error=RuntimeError("storage unavailable"))

    with client_for(monkeypatch, tmp_path, store=store) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("document.pdf", pdf_path.read_bytes(), "application/pdf")},
        )

    assert response.status_code == 422
    assert "storage unavailable" in response.json()["detail"]
