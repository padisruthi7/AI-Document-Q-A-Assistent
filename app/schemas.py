from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None


class Source(BaseModel):
    filename: str
    page_number: int
    chunk_index: int


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool


class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    pages_processed: int
    chunks_created: int
