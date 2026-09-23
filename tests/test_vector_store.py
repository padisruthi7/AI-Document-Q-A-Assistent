from app.services.vector_store import VectorStore


def test_vector_store_adds_and_retrieves_chunks(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    chunks = [
        {"text": "Python is used for data analysis.", "metadata": {"page_number": 2}},
        {"text": "FastAPI provides HTTP endpoints.", "metadata": {"page_number": 4}},
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    store.add_chunks("document-1", "guide.pdf", chunks, embeddings)
    results = store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == chunks[0]["text"]
    assert results[0]["metadata"]["filename"] == "guide.pdf"
    assert results[0]["metadata"]["page_number"] == 2
