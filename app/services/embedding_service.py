import logging
import time
from typing import Sequence

import httpx

from app.core.config import EMBEDDING_MODEL_DIMENSIONS, get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        # OpenAI allows up to 2048 inputs per request
        # Use larger batches for small files, smaller for reliability on large files
        self.max_batch_size = 2048  # OpenAI's maximum
        self.default_batch_size = 500  # Good balance for most files

    def get_dimension(self, model: str) -> int:
        return EMBEDDING_MODEL_DIMENSIONS.get(model, EMBEDDING_MODEL_DIMENSIONS[self.settings.default_embedding_model])

    def validate_dimension(self, model: str) -> None:
        expected = self.settings.vector_dimension
        actual = self.get_dimension(model)
        if actual != expected:
            raise ValueError(
                f"Embedding model dimension {actual} does not match configured vector dimension {expected}"
            )

    async def embed_texts(self, texts: Sequence[str], model: str, batch_size: int | None = None) -> list[list[float]]:
        self.validate_dimension(model)
        if not texts:
            return []
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        
        texts_list = list(texts)
        total_texts = len(texts_list)
        
        # Optimize batch size based on number of texts
        if batch_size is None:
            if total_texts <= 50:
                # Very small files: send all at once (up to max)
                batch_size = min(total_texts, self.max_batch_size)
            elif total_texts <= 500:
                # Small-medium files: use larger batches for speed
                batch_size = min(500, self.max_batch_size)
            else:
                # Large files: use default batch size for reliability
                batch_size = self.default_batch_size
        
        # If all texts fit in one batch, process immediately (fastest for small files)
        if total_texts <= batch_size:
            logger.debug(f"Processing {total_texts} texts in single batch (small file optimization)")
            return await self._embed_batch(texts_list, model)
        
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
            
            embeddings = await self._embed_batch(batch, model)
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
    
    async def _embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed a single batch of texts."""
        payload = {"input": texts, "model": model}
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        
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
