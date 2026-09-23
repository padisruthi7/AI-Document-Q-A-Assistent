# AI Document Q&A Assistant

A local-first web application for asking grounded questions about uploaded PDF documents. The application extracts PDF text, splits it into chunks, creates Gemini embeddings, stores those chunks in ChromaDB, retrieves relevant passages, and asks Gemini to answer using the retrieved context.

## Features

- Upload PDF documents through a FastAPI endpoint
- Extract text page by page with `pypdf`
- Preserve page metadata for source references
- Split document text into overlapping chunks
- Generate document and question embeddings with Gemini
- Store and search embeddings with ChromaDB
- Generate grounded answers with Gemini
- Return source filename, page number, and chunk index
- Refuse unsupported questions with an unavailable-information response
- Validate file type, file size, and question length
- Browser interface built with HTML, CSS, and JavaScript

## Architecture

```text
Browser
  -> FastAPI upload endpoint
  -> PDF extraction with pypdf
  -> text chunking
  -> Gemini document embeddings
  -> ChromaDB

Browser question
  -> Gemini question embedding
  -> ChromaDB similarity search
  -> relevance threshold filtering
  -> grounded Gemini prompt
  -> answer and sources
```

## Technology Stack

- Python 3.12
- FastAPI
- Uvicorn
- Google Gemini through `google-genai`
- ChromaDB
- pypdf
- HTML, CSS, and JavaScript
- Pytest

## Project Structure

```text
.
├── app/
│   ├── main.py                 # FastAPI application and endpoints
│   ├── config.py               # Environment variables and local paths
│   ├── schemas.py              # API request and response models
│   └── services/
│       ├── documents.py        # PDF extraction and chunking
│       ├── gemini.py           # Gemini embeddings and generation
│       └── vector_store.py     # ChromaDB storage and retrieval
├── frontend/
│   ├── index.html              # Browser interface
│   ├── styles.css              # Interface styling
│   └── app.js                  # Browser API requests and display logic
├── tests/
│   ├── test_documents.py       # Extraction and chunking tests
│   ├── test_documents_api.py   # Upload endpoint tests
│   ├── test_questions.py       # Grounding and retrieval tests
│   └── test_vector_store.py    # ChromaDB tests
├── .env.example                # Safe configuration template
├── .gitignore                  # Secrets, local data, and environments to exclude
├── requirements.txt            # Python dependencies
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.12
- A Gemini API key
- Internet access for Gemini API requests

## Local Setup

From the project directory, create and activate a virtual environment.

### Windows PowerShell

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current terminal, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

## Environment Configuration

Create a local `.env` file from the template:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set the values locally:

```env
GEMINI_API_KEY=your_real_key_here
GEMINI_GENERATION_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
RETRIEVAL_DISTANCE_THRESHOLD=0.65
```

Never commit `.env` or share its contents. The API key is loaded by `python-dotenv` in `app/config.py` and is used only by the backend.

## Run Locally

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

Open the application at:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/api/health
```

## API Endpoints

### `GET /api/health`

Returns basic application status and whether the Gemini key is configured. It never returns the key.

### `POST /api/documents`

Uploads and ingests one PDF using `multipart/form-data` with the field name `file`.

Example response:

```json
{
  "message": "Document processed successfully.",
  "document_id": "generated-id",
  "filename": "guide.pdf",
  "pages_processed": 4,
  "chunks_created": 8
}
```

### `POST /api/questions`

Accepts a JSON question:

```json
{
  "question": "What does the document say about storage?"
}
```

Returns an answer, source metadata, and a grounding flag:

```json
{
  "answer": "...",
  "sources": [
    {
      "filename": "guide.pdf",
      "page_number": 2,
      "chunk_index": 3
    }
  ],
  "grounded": true
}
```

When the retrieved context does not support an answer, the application returns an unavailable-information message, `grounded: false`, and no sources.

## How RAG Works

RAG means Retrieval-Augmented Generation. Instead of sending an entire document directly to Gemini:

1. PDF text is extracted page by page.
2. Text is split into smaller chunks.
3. Each chunk is converted into an embedding, which is a numeric representation of meaning.
4. Chunks and metadata are stored in ChromaDB.
5. The user question is converted into an embedding.
6. ChromaDB retrieves the closest chunks.
7. The relevance threshold removes weak matches.
8. Gemini receives the question and retrieved context.
9. The response includes the answer and source metadata.

The system reduces unsupported answers with retrieval filtering and a grounded prompt. No LLM system can guarantee zero hallucinations.

## Local Data

During local development, generated data is stored under `data/`:

- `data/uploads/`: uploaded PDF files
- `data/chroma/`: persistent ChromaDB files

These files are intentionally excluded from Git by `.gitignore`. Do not commit private documents or generated vector data.

## Testing

Run the complete test suite:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

The current suite covers:

- PDF extraction and empty-document handling
- Chunking and page metadata
- ChromaDB insertion and retrieval
- Successful PDF upload
- Invalid file extension
- Oversized upload
- Gemini/service failure handling
- Storage failure handling
- Grounded and unsupported question behavior
- Retrieval threshold filtering
- The annual-revenue grounding regression

The latest verified result is 15 passing tests. The suite currently reports dependency deprecation warnings from the test client stack; they do not fail the tests.

## Security Notes

This is a local MVP and is not a production-secure multi-user application.

Current protections:

- API key is loaded from `.env`, not source code
- `.env` is excluded by `.gitignore`
- Upload size is limited to 10 MB
- PDF filename extension is checked
- Generated data is excluded from Git

Current limitations:

- No authentication or authorization
- No per-user document isolation
- CORS allows all origins
- Uploaded content is stored locally
- Prompt injection and malicious document content require additional controls
- Scanned PDFs requiring OCR are not supported

## GitHub Readiness

Before creating a GitHub repository, verify that these are not tracked or staged:

```powershell
git status --short
Get-ChildItem -Force .env
```

The repository should exclude:

- `.env`
- `venv/`
- `data/`
- uploaded PDFs
- ChromaDB files
- caches and local editor files

A Git repository has not yet been initialized in this project.

## Deployment Status

The current architecture is suitable for local development only. Vercel deployment requires additional design work because local filesystem uploads and persistent ChromaDB storage are not reliable in serverless functions.

Before deployment, the project needs a hosted or external document/vector storage strategy, deployment-specific API routing, environment-variable configuration, and production security controls.

## Known Limitations and Future Improvements

- Add authentication and document ownership
- Add document listing and deletion
- Add OCR for scanned PDFs
- Add retries, timeouts, and structured logging
- Add broader negative and integration tests
- Add conversation history if needed
- Add deployment-compatible persistent storage
- Add monitoring and answer-quality evaluation

## License

No license has been selected yet.
