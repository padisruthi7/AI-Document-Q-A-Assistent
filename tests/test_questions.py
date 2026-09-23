from fastapi.testclient import TestClient

import app.main as main


class FakeService:
    def __init__(self, answer):
        self.answer_text = answer

    def embed_query(self, question):
        return [1.0, 0.0]

    def answer(self, question, matches):
        return self.answer_text


class FakeStore:
    def __init__(self, matches):
        self.matches = matches
        self.seen_embedding = None

    def search(self, embedding, top_k, document_id=None):
        self.seen_embedding = embedding
        return self.matches


def match(distance=0.4):
    return {
        "text": "The document stores chunks in ChromaDB.",
        "distance": distance,
        "metadata": {"filename": "guide.pdf", "page_number": 1, "chunk_index": 0},
    }


def request_with(monkeypatch, matches, answer, threshold=0.65):
    store = FakeStore(matches)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "ai_service", lambda: FakeService(answer))
    monkeypatch.setattr(main, "RETRIEVAL_DISTANCE_THRESHOLD", threshold)
    with TestClient(main.app) as client:
        response = client.post("/api/questions", json={"question": "What does the document say?"})
    return response


def test_supported_question_returns_grounded_sources(monkeypatch):
    response = request_with(monkeypatch, [match()], "The chunks are stored in ChromaDB.")

    assert response.status_code == 200
    assert response.json()["grounded"] is True
    assert response.json()["sources"] == [
        {"filename": "guide.pdf", "page_number": 1, "chunk_index": 0}
    ]


def test_unsupported_question_returns_no_sources(monkeypatch):
    fallback = "The uploaded documents do not contain enough information to answer this question."
    response = request_with(monkeypatch, [match()], fallback)

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []


def test_supported_answer_variants_are_still_treated_as_unavailable(monkeypatch):
    response = request_with(
        monkeypatch,
        [match()],
        "The uploaded documents do not provide enough information to answer this question.",
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []


def test_matches_above_threshold_are_excluded(monkeypatch):
    response = request_with(monkeypatch, [match(0.4), match(0.8)], "A grounded answer.", threshold=0.5)

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1


def test_configured_threshold_is_used(monkeypatch):
    response = request_with(monkeypatch, [match(0.6)], "A grounded answer.", threshold=0.5)

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []


def test_annual_revenue_regression_has_no_grounded_source(monkeypatch):
    fallback = "The uploaded documents do not contain enough information to answer this question."
    response = request_with(monkeypatch, [match(0.423)], fallback)

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []

def test_conversation_history_is_passed_to_service(monkeypatch):
    captured_history = None

    class HistoryService:
        def embed_query(self, question):
            return [1.0, 0.0]

        def answer(self, question, matches, history=None):
            nonlocal captured_history
            captured_history = history
            return "Follow-up answer."

    monkeypatch.setattr(main, "ai_service", lambda: HistoryService())
    monkeypatch.setattr(main, "store", FakeStore([match()]))

    history = [
        {"role": "user", "content": "What are my technical skills?"},
        {"role": "assistant", "content": "Python, Java, and Generative AI skills."},
    ]

    with TestClient(main.app) as client:
        response = client.post(
            "/api/questions",
            json={
                "question": "Which of these are related to Generative AI?",
                "history": history,
            },
        )

    assert response.status_code == 200
    assert captured_history == history