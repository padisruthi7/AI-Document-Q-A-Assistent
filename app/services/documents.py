from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pages(path: Path) -> list[PageText]:
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page_number=index, text=text))
    if not pages:
        raise ValueError("The PDF contains no extractable text.")
    return pages


def chunk_pages(pages: list[PageText], chunk_size: int = 900, overlap: int = 120) -> list[dict]:
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    chunks = []
    for page in pages:
        words = re.findall(r"\S+", page.text)
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append({
                "text": " ".join(words[start:end]),
                "metadata": {"page_number": page.page_number},
            })
            if end == len(words):
                break
            start = end - overlap
    return chunks
