from fastapi.testclient import TestClient

import app.main as main
from app.services.vector_store import VectorStore


class FakeService:
    def __init__(self, answers=None):
        self.answers = answers or {}

    def embed_query(self, question):
        return [1.0, 0.0] if "Aurora" in question else [0.9, 0.1]

    def answer(self, question, matches):
        return " | ".join(match["text"] for match in matches)


def document_chunks(document_id, filename, text):
    return [{"text": text, "metadata": {"page_number": 1}}]


def test_two_documents_are_stored_with_distinct_ids(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.add_chunks(
        "document-a",
        "aurora.pdf",
        document_chunks("document-a", "aurora.pdf", "Project Aurora has a budget of 10 million dollars."),
        [[1.0, 0.0]],
    )
    store.add_chunks(
        "document-b",
        "beacon.pdf",
        document_chunks("document-b", "beacon.pdf", "Project Beacon has a budget of 25 million dollars."),
        [[0.9, 0.1]],
    )

    records = store.collection.get()

    assert set(records["ids"]) == {"document-a-0", "document-b-0"}
    assert {metadata["document_id"] for metadata in records["metadatas"]} == {
        "document-a",
        "document-b",
    }


def test_question_about_each_document_retrieves_its_best_match(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.add_chunks(
        "document-a",
        "aurora.pdf",
        document_chunks("document-a", "aurora.pdf", "Project Aurora has a budget of 10 million dollars."),
        [[1.0, 0.0]],
    )
    store.add_chunks(
        "document-b",
        "beacon.pdf",
        document_chunks("document-b", "beacon.pdf", "Project Beacon has a budget of 25 million dollars."),
        [[0.0, 1.0]],
    )

    aurora = store.search([1.0, 0.0], top_k=1)
    beacon = store.search([0.0, 1.0], top_k=1)

    assert aurora[0]["metadata"]["document_id"] == "document-a"
    assert "10 million" in aurora[0]["text"]
    assert beacon[0]["metadata"]["document_id"] == "document-b"
    assert "25 million" in beacon[0]["text"]


def test_search_is_global_and_can_return_a_similar_other_document(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.add_chunks(
        "document-a",
        "aurora.pdf",
        document_chunks("document-a", "aurora.pdf", "Project Aurora has a budget of 10 million dollars."),
        [[1.0, 0.0]],
    )
    store.add_chunks(
        "document-b",
        "beacon.pdf",
        document_chunks("document-b", "beacon.pdf", "Project Beacon has a budget of 25 million dollars."),
        [[0.99, 0.01]],
    )

    results = store.search([1.0, 0.0], top_k=2)
    document_ids = [result["metadata"]["document_id"] for result in results]

    assert set(document_ids) == {"document-a", "document-b"}


def test_question_api_filters_to_document_a(monkeypatch):
    class RecordingStore:
        def __init__(self):
            self.document_id = None

        def search(self, embedding, top_k, document_id=None):
            self.document_id = document_id
            return [
                {
                    "text": "Project Aurora has a budget of 10 million dollars.",
                    "distance": 0.1,
                    "metadata": {"filename": "aurora.pdf", "page_number": 1, "chunk_index": 0},
                }
            ] if document_id == "document-a" else []

    store = RecordingStore()
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "ai_service", lambda: FakeService())

    with TestClient(main.app) as client:
        response = client.post(
            "/api/questions",
            json={"question": "What is the Aurora budget?", "document_id": "document-a"},
        )

    assert response.status_code == 200
    assert response.json()["sources"][0]["filename"] == "aurora.pdf"
    assert store.document_id == "document-a"


def test_question_api_filters_to_document_b(monkeypatch):
    class RecordingStore:
        def search(self, embedding, top_k, document_id=None):
            assert document_id == "document-b"
            return [{
                "text": "Project Beacon has a budget of 25 million dollars.",
                "distance": 0.1,
                "metadata": {"filename": "beacon.pdf", "page_number": 1, "chunk_index": 0},
            }]

    monkeypatch.setattr(main, "store", RecordingStore())
    monkeypatch.setattr(main, "ai_service", lambda: FakeService())

    with TestClient(main.app) as client:
        response = client.post(
            "/api/questions",
            json={"question": "What is the Beacon budget?", "document_id": "document-b"},
        )

    assert response.status_code == 200
    assert response.json()["sources"][0]["filename"] == "beacon.pdf"


def test_nonexistent_document_id_returns_unavailable_response(monkeypatch):
    class EmptyStore:
        def search(self, embedding, top_k, document_id=None):
            assert document_id == "missing-document"
            return []

    monkeypatch.setattr(main, "store", EmptyStore())
    monkeypatch.setattr(main, "ai_service", lambda: FakeService())

    with TestClient(main.app) as client:
        response = client.post(
            "/api/questions",
            json={"question": "What is the budget?", "document_id": "missing-document"},
        )

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []


def test_question_api_without_document_id_preserves_global_search(monkeypatch):
    class RecordingStore:
        def __init__(self):
            self.document_id = "sentinel"

        def search(self, embedding, top_k, document_id=None):
            self.document_id = document_id
            return [{
                "text": "Global result.",
                "distance": 0.1,
                "metadata": {"filename": "global.pdf", "page_number": 1, "chunk_index": 0},
            }]

    store = RecordingStore()
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "ai_service", lambda: FakeService())

    with TestClient(main.app) as client:
        response = client.post("/api/questions", json={"question": "What is the budget?"})

    assert response.status_code == 200
    assert store.document_id is None
    assert response.json()["sources"][0]["filename"] == "global.pdf"


def test_document_listing_returns_unique_documents(monkeypatch):
    class ListingStore:
        def list_documents(self):
            return [
                {"document_id": "doc-a", "filename": "aurora.pdf"},
                {"document_id": "doc-b", "filename": "beacon.pdf"},
            ]

    monkeypatch.setattr(main, "store", ListingStore())

    with TestClient(main.app) as client:
        response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == [
        {"document_id": "doc-a", "filename": "aurora.pdf"},
        {"document_id": "doc-b", "filename": "beacon.pdf"},
    ]
