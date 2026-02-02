## Local embedding dependencies (optional)

The default app and Docker image **do not** install `sentence-transformers`, because it pulls in PyTorch (~1–2GB) and can cause very long or failing Docker builds (I/O errors, timeouts). To use **local embeddings** (`USE_LOCAL_EMBEDDINGS=true`):

- **Local / venv:** `pip install -r requirements-local-embeddings.txt`
- **Docker:** Either build a custom image that runs `pip install -r requirements-local-embeddings.txt` after the base `requirements.txt`, or run that command in the container after startup. See README for `.env` and `VECTOR_DIMENSION` setup.

---

## Chunks vs embeddings vs vectors

| Term | What it is | Example |
|------|------------|--------|
| **Chunk** | A **piece of text** split from a document (e.g. 300 words with 50-word overlap). Human-readable. | *"To change your login credentials, go to Settings → Account → Security. Click Reset password and follow the email link."* |
| **Vector** | A **list of numbers** (e.g. 384 or 1536 floats) that represents that text for similarity search. Not human-readable. | `[0.023, -0.141, 0.891, ...]` |
| **Embedding** | The **vector you get** when you pass text (a chunk or a query) through the embedding model. So: *embedding = the output vector*. In practice we often say "embedding" and "vector" for the same thing. | Same as vector: the list of numbers produced by the model. |

**In one line:** A **chunk** is the text; the **embedding** (or **vector**) is the list of numbers that represents that text so we can compare "meaning" by comparing vectors.

**Flow:** Chunk (text) → embedding model → **embedding** (= **vector**) → store in pgvector. At search: query (text) → same model → query **embedding** (= query **vector**) → find stored **vectors** closest to it → return the **chunks** (text) that produced those vectors.

---

What the embedding model is for
The embedding model turns text into vectors (lists of numbers). Those vectors are what you store in pgvector and compare with the user’s query. So:
At ingest: chunk text → embedding model → vector → store in pgvector
At search: user query → same embedding model → query vector → search pgvector for nearest vectors → get matching chunks
At chat: matching chunks + user query → ChatGPT API → answer
So the embedding model is used in two places: when creating embeddings for chunks, and when creating the embedding for the user query. Both must use the same model (and same dimension) so that “similar meaning” = “similar vectors”.


Why we need it (purpose)
Without an embedding model you only have raw text. You can’t do “find chunks that are similar in meaning to this question” in a principled way.
Keyword search (e.g. SQL LIKE or BM25):
Matches words, not meaning.
Query: “How do I reset my password?”
Chunk: “To change your login credentials, go to Settings…”
May not match well even though it’s the right answer.
Embedding-based search (what you do now):
Query and chunks are turned into vectors by the embedding model.
“Reset password” and “change login credentials” get similar vectors because the model encodes meaning.
pgvector then finds chunks whose vectors are closest to the query vector → you get the most semantically relevant chunks, not just keyword overlap.
So the purpose of the embedding model is: to convert text into vectors so we can do semantic similarity search (meaning-based retrieval). That’s the “R” in RAG: retrieval by meaning, not just keywords.
End-to-end flow (how it fits)
Ingest:
Document → chunks → embedding model → one vector per chunk → store vectors (and text) in pgvector.
Search:
User question → same embedding model → one vector for the query → pgvector “nearest vectors” search → list of most relevant chunks (by meaning).
Chat:
Those chunks + user question → sent to ChatGPT (or another LLM) → model uses that context to generate the answer.
So: we use the embedding model to create the embeddings we store in pgvector and to create the query embedding we search with. Its purpose is to enable semantic search so that the chunks sent to the ChatGPT API are the ones that actually match what the user is asking, not just the words they used.

---

## Local embedding model + ChatGPT API: how they work together

When you use a **local embedding model** (e.g. sentence-transformers) and **ChatGPT** (API) together, each has a clear role. Both support the same RAG scenario; only *where* the work runs changes.

### What each process does

| Component | Role | Where it runs | What it does |
|-----------|------|----------------|--------------|
| **Local embedding model** | **Retrieval** (R in RAG) | Your server (API process) | Converts text → vectors. Used at **ingest** (chunks → vectors → pgvector) and at **search** (user query → vector → similarity search). No API key, no network. |
| **ChatGPT (API)** | **Generation** (G in RAG) | OpenAI’s servers | Takes **retrieved chunks + user question** as context and generates the answer. Needs `OPENAI_API_KEY` and network. |

### End-to-end process (local embeddings + ChatGPT)

1. **Ingest (one-time per document)**  
   - Document → chunks → **local embedding model** (on your server) → vectors → stored in pgvector.  
   - No call to OpenAI here.

2. **User asks a question**  
   - User query → **same local embedding model** (on your server) → query vector.  
   - pgvector finds chunks whose vectors are closest to this query vector → you get the most relevant chunks.  
   - Still no call to OpenAI.

3. **Generate answer**  
   - You send **only** the retrieved chunk texts + user question to the **ChatGPT API**.  
   - ChatGPT does not see any vectors; it only sees text (chunks + question) and produces the answer.

So: **local model = retrieval (embedding + search)**; **ChatGPT = generation (answer from context)**. They support the scenario by splitting “find the right chunks” (local) and “write the answer from those chunks” (API).

### Requirements they support

- **Same embedding model for ingest and search**  
  Use one local model (e.g. `all-MiniLM-L6-v2`) for both chunk embeddings and query embeddings. Same model → comparable vectors → correct similarity search.

- **Same vector dimension everywhere**  
  Set `VECTOR_DIMENSION` to match your local model (e.g. 384). pgvector and your app must agree on this.

- **ChatGPT only needs text**  
  You pass the *text* of the retrieved chunks and the user’s question. No vectors go to ChatGPT; it doesn’t need to know about embeddings.

- **Cost and privacy**  
  - Local embeddings: no per-request embedding cost; data stays on your server for retrieval.  
  - ChatGPT: you pay per token for the generation call; the chunks + question are sent to OpenAI for that step only.

### Summary

- **Local embedding model:** runs on your server, handles all “text → vector” and “similarity search.” Supports the requirement: “find chunks that match the user’s question by meaning.”
- **ChatGPT:** runs on OpenAI, handles “context + question → answer.” Supports the requirement: “generate a natural-language answer from those chunks.”
- Together they support the full RAG flow: **retrieve** with the local model, **generate** with ChatGPT.