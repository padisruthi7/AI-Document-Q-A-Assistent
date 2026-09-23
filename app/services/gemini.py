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

    

    def answer(
        self, question: str, matches: list[dict], history: list[dict[str, str]] | None = None,) -> str:
        history = history or []
        context = "\n\n".join(
            f"Source: {item['metadata']['filename']}, page {item['metadata']['page_number']}\n{item['text']}"
            for item in matches
        )

        conversation = "\n".join(
    f"{message['role'].capitalize()}: {message['content']}"
                for message in history
                if message.get("role") in {"user", "assistant"} and message.get("content")
        )

        prompt = f"""You answer questions about uploaded documents.
Use only the document context below to answer the current question.
Conversation history is provided only to understand references to earlier questions or answers.
Do not use outside knowledge.

If the answer is not supported by the document context, say exactly:
The uploaded documents do not contain enough information to answer this question.

Keep the answer concise and factual.

Conversation history:
{conversation or "No previous conversation."}

Document context:
{context}

Current question: {question}
"""
        response = self.client.models.generate_content(
        model=GEMINI_GENERATION_MODEL,
        contents=prompt,
        ) 
        return response.text.strip()
