# MedQuery RAG – Project Documentation (Summary)

## 1. What the project does (start to end)

**MedQuery** is a multi-tenant **RAG (Retrieval-Augmented Generation)** API. It:

1. **Ingests** documents (PDF/TXT) per business/workspace: extract text → chunk → embed → store vectors in PostgreSQL (pgvector).
2. **Retrieves** relevant chunks for a user query: embed query → similarity search in pgvector → return top chunks.
3. **Chat** (RAG + LLM): same retrieval as above, then send **query + retrieved chunks + system prompt** to an LLM (OpenAI) and return the generated answer.

All data is scoped by **business_client_id** and **workspace_id** (multi-tenant).

---

## 2. Models used

| Purpose | Model | Where configured | When used |
|--------|--------|-------------------|-----------|
| **Embeddings (vectors)** | **Local:** `sentence-transformers/all-MiniLM-L6-v2` (384 dim) | App config / .env: `local_embedding_model` | When workspace has `use_local_embeddings = true` |
| **Embeddings (vectors)** | **OpenAI:** `text-embedding-3-small` (1536 dim) or workspace `embedding_model` | Workspace config in DB | When workspace has `use_local_embeddings = false` |
| **Chat (answer generation)** | **OpenAI:** e.g. `gpt-4.1-mini` | Workspace config: `chat_model_default` | Always (no local chat model) |

- **Embedding model** is chosen **per workspace** (DB: `workspace_config.use_local_embeddings` and `embedding_model`).
- **Chat model** is from workspace config (`chat_model_default`), with optional overrides from the client.

---

## 3. Embedding process (detailed)

### 3.1 What embeddings are

- Text (a chunk or a query) is turned into a **vector**: a fixed-length list of numbers (e.g. 384 or 1536).
- **Vector dimension** (e.g. **384**) = length of that list. It must match the embedding model and the DB column.

### 3.2 Where embeddings happen

- **At ingest:** Each document chunk is embedded and stored in `document_chunks.embedding` (pgvector column).
- **At query (RAG / Chat):** The user query is embedded once; that vector is used for similarity search.

### 3.3 Which model is used (per workspace)

- **`use_local_embeddings = true`** (workspace config in DB):
  - Model: **`local_embedding_model`** from app config → `sentence-transformers/all-MiniLM-L6-v2`.
  - Output dimension: **384**.
  - No API call; runs in-process (sentence-transformers).

- **`use_local_embeddings = false`** (default for new workspaces):
  - Model: workspace’s **`embedding_model`** (e.g. `text-embedding-3-small`).
  - Output dimension: **1536** for OpenAI small (must match DB if you use OpenAI embeddings).
  - Uses **OpenAI Embeddings API**; requires **OpenAI API key** (from DB or .env).

### 3.4 Vector dimension and DB

- **`vector_dimension`** in config/.env (e.g. **384**) defines:
  - The **pgvector column size** in `document_chunks` (e.g. `vector(384)`).
  - The **expected** embedding size; the app validates that the chosen embedding model’s output matches this.
- If you use **local** embeddings (384), DB must be 384. If you switch to OpenAI (1536), you need a migration to change the column to 1536 and re-ingest.

### 3.5 Embedding flow in code

- **Ingest:** `RagIngestService` → chunks text → `EmbeddingService.embed_texts(...)` with workspace `use_local` and `embedding_model` → vectors written to `document_chunks.embedding`.
- **Chat / Retrieve:** Query text → `EmbeddingService.embed_texts([query], ...)` with same workspace settings → one query vector → used in pgvector similarity search.

---

## 4. End-to-end flows

### 4.1 Document ingestion (RAG build)

1. Upload document (PDF/TXT) for a business/workspace.
2. Extract text (PDF: page-by-page to limit memory).
3. Chunk text (words per chunk, overlap from workspace config).
4. Get **OpenAI API key** from DB (`system_config`) or .env fallback.
5. For each batch of chunks: **embed** (local or OpenAI per workspace) → insert into `document_chunks` (with `embedding` vector).
6. Document marked indexed; chunks are searchable by vector.

### 4.2 Chat (user query → answer)

1. User sends message in Chat UI (business, workspace, query, system prompt).
2. **API** resolves business/workspace, loads **workspace config** from DB.
3. **OpenAI API key:** from DB (`system_config`) first, else .env.
4. **Embed query:** local or OpenAI embedding model per workspace → **query vector**.
5. **RAG retrieval:** pgvector similarity search (cosine) for that business/workspace → **top_k** chunks (configurable).
6. **Prompt:** System message = user’s prompt engineering; user message = “CONTEXT: {retrieved chunks}\n\nQUESTION: {query}”.
7. **LLM:** Call OpenAI Chat API (e.g. gpt-4.1-mini) with that prompt → **answer**.
8. Save **chat_request** and **chat_response** in DB; return **answer**, **sources**, **usage** to UI.

### 4.3 Retrieve-only (no LLM)

- Same as chat up to and including **embed query** and **RAG retrieval**.
- Returns **retrieved chunks** (and scores) only; no ChatGPT call.

---

## 5. Configuration (where things live)

| What | Where | Scope |
|------|--------|--------|
| **OpenAI API key** | DB table **`system_config`** (key `openai_api_key`), editable in **System configurations** UI; fallback: .env | Global |
| **Vector dimension** | .env / config: **`vector_dimension`** (e.g. 384) | Global |
| **Local embedding model** | Config / .env: **`local_embedding_model`** (e.g. all-MiniLM-L6-v2) | Global |
| **Use local embeddings** | DB: **`workspace_config.use_local_embeddings`** | Per workspace |
| **Embedding model (API)** | DB: **`workspace_config.embedding_model`** (e.g. text-embedding-3-small) | Per workspace |
| **Chat model, temperature, max_tokens** | DB: **`workspace_config`** (chat_model_default, etc.) | Per workspace |
| **Prompt engineering (system prompt)** | DB: **`workspace_config.prompt_engineering`** | Per workspace |
| **RAG settings** (top_k, similarity_threshold, chunk size) | DB: **`workspace_config`** | Per workspace |

---

## 6. Logging

- **`logs/chat.log`:** Per-request chat flow (query received, embedding, retrieval, prompt, OpenAI call, response, errors). One JSON line per step.
- **`logs/ingestion.log`:** Per-document ingest steps (extract, chunk, embed batches, commit, memory). One JSON line per step.
- **`logs/crashes/`:** Crash and error reports (e.g. OOM, exceptions). See **`logs/README.md`**.

---

## 7. Summary table

| Stage | Input | Output | Model / API |
|-------|--------|--------|-------------|
| Ingest: extract | PDF/TXT file | Raw text | PyMuPDF / pypdf |
| Ingest: chunk | Raw text | Text chunks | Config (chunk_words, overlap) |
| Ingest: embed | Chunks | Vectors | Local (MiniLM) or OpenAI Embeddings |
| Ingest: store | Vectors + metadata | Rows in `document_chunks` | PostgreSQL + pgvector |
| Chat: embed query | User query | Query vector | Same as ingest (per workspace) |
| Chat: retrieve | Query vector | Top-k chunks | pgvector similarity search |
| Chat: prompt | Chunks + query + system prompt | Messages for LLM | — |
| Chat: generate | Messages | Answer text | OpenAI Chat (e.g. gpt-4.1-mini) |

This document is a concise reference for what the project does, which models it uses, how embedding works, and how configuration and logging are organized.
