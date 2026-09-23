from pathlib import Path

import chromadb


class VectorStore:
    def __init__(self, storage_path: Path):
        client = chromadb.PersistentClient(path=str(storage_path))
        self.collection = client.get_or_create_collection(
            name="document_chunks", metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, document_id, filename, chunks, embeddings):
        self.collection.add(
            ids=[f"{document_id}-{i}" for i in range(len(chunks))],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {"filename": filename, "document_id": document_id,
                 "page_number": chunk["metadata"]["page_number"], "chunk_index": i}
                for i, chunk in enumerate(chunks)
            ],
        )

    def list_documents(self):
        result = self.collection.get(include=["metadatas"])
        grouped = {}
        for metadata in result.get("metadatas", []):
            if not metadata:
                continue
            document_id = metadata.get("document_id")
            filename = metadata.get("filename")
            if not document_id or document_id in grouped:
                continue
            grouped[document_id] = {"document_id": document_id, "filename": filename}
        return list(grouped.values())

    def search(self, embedding, top_k=5, document_id=None):
        query_options = {"query_embeddings": [embedding], "n_results": top_k}
        if document_id is not None:
            query_options["where"] = {"document_id": document_id}
        result = self.collection.query(**query_options)
        return [
            {"text": text, "metadata": metadata, "distance": distance}
            for text, metadata, distance in zip(
                result.get("documents", [[]])[0],
                result.get("metadatas", [[]])[0],
                result.get("distances", [[]])[0],
            )
        ]
