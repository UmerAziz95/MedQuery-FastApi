# MedQuery RAG API

FastAPI-only backend for a multi-tenant, medical RAG platform using PostgreSQL + pgvector.

## Features
- Multi-tenant isolation by `business_client_id` + `workspace_id`.
- Admin JWT authentication (optional super admin).
- Workspace configuration for chunking, retrieval, and chat defaults.
- Document ingestion (PDF/TXT), chunking, embeddings, pgvector storage.
- Retrieval-only and chat endpoints.
- Full Swagger/OpenAPI docs at `/docs`.

## Setup

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment
Create a `.env` file:
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/medquery
JWT_SECRET_KEY=change-me
OPENAI_API_KEY=sk-...
VECTOR_DIMENSION=1536
```

### 3) Database and migrations
```bash
alembic upgrade head
```

### 4) Run the API
```bash
uvicorn app.main:app --reload
```

### 5) Open the test UI
Visit `http://localhost:8000/ui` for a ChatGPT-style console that targets the existing
`/api/chat/generate` and `/api/rag/retrieve` endpoints. Provide a bearer token if your
workspace requires authentication.

## Docker setup

### 1) Configure environment
Create a `.env` file in the repo root (same variables as above):
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/medquery
JWT_SECRET_KEY=change-me
OPENAI_API_KEY=sk-...
VECTOR_DIMENSION=1536
```

### 2) Start services
```bash
docker compose up --build
```

### 3) Run migrations
```bash
docker compose exec api alembic upgrade head
```

### 4) Open the UI
Visit `http://localhost:8000/ui` to use the testing console (same endpoints consumed by third parties).

## API Walkthrough

### Bootstrap a super admin
If no admins exist, `POST /api/admin/auth/create-admin` can be called without a token:
```json
{
  "business_client_id": "acme",
  "email": "admin@acme.com",
  "password": "secret",
  "role": "super_admin"
}
```

### Login
```json
POST /api/admin/auth/login
{
  "business_client_id": "acme",
  "email": "admin@acme.com",
  "password": "secret"
}
```

### Create business
```json
POST /api/admin/businesses
{
  "business_client_id": "acme",
  "name": "Acme Health"
}
```

### Create workspace
```json
POST /api/admin/businesses/acme/workspaces
{
  "workspace_id": "main",
  "name": "Main Workspace"
}
```

### Update workspace config
```json
PUT /api/admin/businesses/acme/workspaces/main/config
{
  "chunk_words": 300,
  "overlap_words": 50,
  "top_k": 5,
  "similarity_threshold": 0.2,
  "max_context_chars": 12000,
  "embedding_model": "text-embedding-3-small",
  "chat_model_default": "gpt-4.1-mini",
  "chat_temperature_default": 0.2,
  "chat_max_tokens_default": 600
}
```

### Upload and index documents
```bash
curl -X POST \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@guide.pdf" \
  "http://localhost:8000/api/admin/businesses/acme/workspaces/main/documents/upload"
```

### Retrieve chunks
```json
POST /api/rag/retrieve
{
  "business_client_id": "acme",
  "workspace_id": "main",
  "user_id": "u123",
  "query": "What are symptoms of ...?",
  "top_k": 5
}
```

### Generate chat response
```json
POST /api/chat/generate
{
  "business_client_id": "acme",
  "workspace_id": "main",
  "user_id": "u123",
  "query": "User question here",
  "prompt_engineering": "You are a medical assistant. Provide concise answers.",
  "chat_config_override": {
    "model": "gpt-4.1-mini",
    "temperature": 0.2,
    "max_tokens": 600
  }
}
```

## Notes
- `pgvector` extension is created via Alembic migration.
- Document deletion cascades to chunks.
- All retrieval operations are filtered by `business_id` and `workspace_id`.
- `VECTOR_DIMENSION` must match the embedding model dimension used in workspace configs.
