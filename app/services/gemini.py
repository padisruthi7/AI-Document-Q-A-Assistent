from google import genai

from app.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_GENERATION_MODEL


class GeminiService:
    def __init__(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env before using AI features.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        return [item.values for item in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return response.embeddings[0].values

    def answer(self, question: str, matches: list[dict]) -> str:
        context = "\n\n".join(
            f"Source: {item['metadata']['filename']}, page {item['metadata']['page_number']}\n{item['text']}"
            for item in matches
        )
        prompt = f"""You answer questions about uploaded documents.
Use only the context below. If the answer is not supported by the context, say exactly:
The uploaded documents do not contain enough information to answer this question.
Do not use outside knowledge. Keep the answer concise and factual.

Context:
{context}

Question: {question}
"""
        response = self.client.models.generate_content(
            model=GEMINI_GENERATION_MODEL,
            contents=prompt,
        )
        return response.text.strip()
