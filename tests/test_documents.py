from pypdf import PdfWriter

from app.services import documents
from app.services.documents import chunk_pages, extract_pages


def create_pdf(path):
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(path)


def test_empty_pdf_is_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    create_pdf(path)

    try:
        extract_pages(path)
    except ValueError as error:
        assert "no extractable text" in str(error)
    else:
        raise AssertionError("Expected an empty PDF to be rejected")


def test_extract_pages_preserves_page_numbers(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("first page"), FakePage("second page")]

    monkeypatch.setattr(documents, "PdfReader", FakeReader)

    pages = extract_pages(tmp_path / "document.pdf")

    assert [(page.page_number, page.text) for page in pages] == [
        (1, "first page"),
        (2, "second page"),
    ]


def test_chunking_preserves_page_metadata():
    pages = [type("Page", (), {"page_number": 3, "text": "one two three four five"})()]

    chunks = chunk_pages(pages, chunk_size=3, overlap=1)

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["page_number"] == 3
    assert chunks[0]["text"] == "one two three"
    assert chunks[1]["text"] == "three four five"
