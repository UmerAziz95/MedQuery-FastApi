import asyncio
import logging
import time
from typing import Sequence

import httpx

from app.core.config import EMBEDDING_MODEL_DIMENSIONS, get_settings

logger = logging.getLogger(__name__)

# Optional local embeddings (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        # Keep batches small to avoid large response payloads and memory spikes.
        self.max_batch_size = 128
        self.default_batch_size = 32
        self._local_model = None

    def _get_local_model(self):
        """Lazy-load the local embedding model (only when use_local_embeddings=True)."""
        if self._local_model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise RuntimeError(
                    "Local embeddings require sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )
            model_name = self.settings.local_embedding_model
            logger.info("Loading local embedding model: %s", model_name)
            self._local_model = SentenceTransformer(model_name)
        return self._local_model

    def get_dimension(self, model: str, use_local: bool | None = None) -> int:
        use_local_flag = use_local if use_local is not None else self.settings.use_local_embeddings
        if use_local_flag:
            dim = EMBEDDING_MODEL_DIMENSIONS.get(
                self.settings.local_embedding_model,
                EMBEDDING_MODEL_DIMENSIONS.get("all-MiniLM-L6-v2", 384),
            )
            return dim
        return EMBEDDING_MODEL_DIMENSIONS.get(model, EMBEDDING_MODEL_DIMENSIONS[self.settings.default_embedding_model])

    def validate_dimension(self, model: str, use_local: bool | None = None) -> None:
        expected = self.settings.vector_dimension
        actual = self.get_dimension(model, use_local=use_local)
        if actual != expected:
            raise ValueError(
                f"Embedding model dimension {actual} does not match configured vector dimension {expected}"
            )

    async def embed_texts(
        self,
        texts: Sequence[str],
        model: str,
        batch_size: int | None = None,
        use_local: bool | None = None,
        openai_api_key: str | None = None,
    ) -> list[list[float]]:
        # Determine if we should use local embeddings: workspace config overrides global setting
        use_local_flag = use_local if use_local is not None else self.settings.use_local_embeddings
        key = openai_api_key or self.settings.openai_api_key
        self.validate_dimension(model, use_local=use_local_flag)
        if not texts:
            return []
        if not use_local_flag and not key:
            raise RuntimeError("OPENAI_API_KEY not configured (set in System configurations or .env, or use local embeddings)")
        
        texts_list = list(texts)
        total_texts = len(texts_list)
        logger.info("Embedding request start: total_texts=%s model=%s use_local=%s", total_texts, model, use_local_flag)
        
        # Optimize batch size based on number of texts, then clamp to safe limits.
        if batch_size is None:
            if total_texts <= 50:
                # Very small files: still cap to safe max.
                batch_size = min(total_texts, self.max_batch_size)
            elif total_texts <= 500:
                # Small-medium files: keep modest to avoid memory spikes.
                batch_size = min(64, self.max_batch_size)
            else:
                # Large files: keep default batch size for reliability
                batch_size = self.default_batch_size
        batch_size = max(1, min(batch_size, self.default_batch_size))
        logger.info("Embedding batch_size selected=%s total_texts=%s", batch_size, total_texts)
        
        # If all texts fit in one batch, process immediately (fastest for small files)
        if total_texts <= batch_size:
            logger.debug(f"Processing {total_texts} texts in single batch (small file optimization)")
            if use_local_flag:
                return await self._embed_batch_local(texts_list)
            return await self._embed_batch(texts_list, model, key)
        
        # Process in batches for large documents
        logger.info(f"Processing {total_texts} texts in batches of {batch_size}")
        all_embeddings = []
        start_time = time.time()
        
        for i in range(0, total_texts, batch_size):
            batch = texts_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_texts + batch_size - 1) // batch_size
            
            if total_batches > 1:
                logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
            batch_start = time.time()
            
            if use_local_flag:
                embeddings = await self._embed_batch_local(batch)
            else:
                embeddings = await self._embed_batch(batch, model, key)
            all_embeddings.extend(embeddings)
            
            batch_time = time.time() - batch_start
            if total_batches > 1:
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / batch_num
                estimated_remaining = avg_time_per_batch * (total_batches - batch_num)
                
                logger.info(
                    f"Batch {batch_num}/{total_batches} completed in {batch_time:.2f}s. "
                    f"Progress: {len(all_embeddings)}/{total_texts} embeddings. "
                    f"Estimated time remaining: {estimated_remaining:.1f}s"
                )
        
        total_time = time.time() - start_time
        logger.info(f"Generated {len(all_embeddings)} embeddings in {total_time:.2f}s ({total_time/total_texts:.3f}s per embedding)")
        
        return all_embeddings

    async def _embed_batch_local(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local sentence-transformers model (no API call)."""
        loop = asyncio.get_event_loop()
        encoder = self._get_local_model()
        # Run encode in thread pool to avoid blocking the event loop
        embeddings = await loop.run_in_executor(
            None,
            lambda: encoder.encode(texts, convert_to_numpy=True).tolist(),
        )
        return embeddings

    async def _embed_batch(self, texts: list[str], model: str, openai_api_key: str | None = None) -> list[list[float]]:
        """Embed a single batch of texts."""
        key = openai_api_key or self.settings.openai_api_key
        payload = {"input": texts, "model": model}
        headers = {"Authorization": f"Bearer {key}"}
        
        # Optimize timeout based on batch size
        # Small batches (< 100): 30s, Medium (100-500): 60s, Large (> 500): 120s
        batch_size = len(texts)
        if batch_size < 100:
            timeout = 30
        elif batch_size < 500:
            timeout = 60
        else:
            timeout = 120
        
        async with httpx.AsyncClient(base_url=self.settings.openai_base_url, timeout=timeout) as client:
            try:
                response = await client.post("/embeddings", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                logger.info("Embedding batch complete: batch_size=%s response_count=%s", len(texts), len(data.get("data", [])))
                return [item["embedding"] for item in data["data"]]
            except httpx.TimeoutException:
                logger.error(f"Timeout embedding batch of {len(texts)} texts")
                raise RuntimeError(f"Embedding request timed out. Try reducing chunk size or batch size.")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error embedding batch: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error embedding batch: {type(e).__name__}: {e}")
                raise
