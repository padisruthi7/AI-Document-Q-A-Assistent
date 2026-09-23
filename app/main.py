import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import CHROMA_DIR, MAX_FILE_SIZE, RETRIEVAL_DISTANCE_THRESHOLD, UPLOAD_DIR
from app.schemas import AnswerResponse, QuestionRequest, Source, UploadResponse
from app.services.documents import chunk_pages, extract_pages
from app.services.gemini import GeminiService
from app.services.vector_store import VectorStore

app = FastAPI(title="AI Document Q&A Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"],)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

store = VectorStore(CHROMA_DIR)


def is_unavailable_answer(answer: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", answer.lower())
    normalized = " ".join(normalized.split())
    if not normalized:
        return False

    unavailable_markers = (
        "not enough information",
        "not contain enough information",
        "not provide enough information",
    )
    question_markers = (
        "answer this question",
        "answer the question",
    )

    if "uploaded documents" in normalized:
        return any(marker in normalized for marker in unavailable_markers) and any(
            marker in normalized for marker in question_markers
        )

    return False


def ai_service():
    try:
        return GeminiService()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("frontend/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "ai_configured": bool(GeminiService.__name__ and __import__('app.config', fromlist=['GEMINI_API_KEY']).GEMINI_API_KEY)}


@app.get("/api/documents")
def list_documents():
    return store.list_documents()


@app.post("/api/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
    document_id = uuid4().hex
    path = UPLOAD_DIR / f"{document_id}.pdf"
    path.write_bytes(content)
    try:
        pages = extract_pages(path)
        chunks = chunk_pages(pages)
        service = ai_service()
        embeddings = service.embed([chunk["text"] for chunk in chunks])
        store.add_chunks(document_id, file.filename, chunks, embeddings)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Document processing failed: {exc}") from exc 
    return UploadResponse(message="Document processed successfully.", document_id=document_id,
                          filename=file.filename, pages_processed=len(pages), chunks_created=len(chunks))


@app.post("/api/questions", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    service = ai_service()
    try:
        matches = store.search(
            service.embed_query(question),
            top_k=5,
            document_id=request.document_id,
        )
        matches = [
            match
            for match in matches
            if match["distance"] <= RETRIEVAL_DISTANCE_THRESHOLD
        ]
        if not matches:
            return AnswerResponse(answer="The uploaded documents do not contain enough information to answer this question.", sources=[], grounded=False)
        answer = service.answer(question, matches)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Question processing failed: {exc}") from exc
    grounded = not is_unavailable_answer(answer)
    sources = [
        Source(
            filename=m["metadata"]["filename"],
            page_number=int(m["metadata"]["page_number"]),
            chunk_index=int(m["metadata"]["chunk_index"]),
        )
        for m in matches
    ] if grounded else []
    return AnswerResponse(answer=answer, sources=sources, grounded=grounded)
