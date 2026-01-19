from typing import Sequence

import httpx

from app.core.config import EMBEDDING_MODEL_DIMENSIONS, get_settings


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def get_dimension(self, model: str) -> int:
        return EMBEDDING_MODEL_DIMENSIONS.get(model, EMBEDDING_MODEL_DIMENSIONS[self.settings.default_embedding_model])

    def validate_dimension(self, model: str) -> None:
        expected = self.settings.vector_dimension
        actual = self.get_dimension(model)
        if actual != expected:
            raise ValueError(
                f"Embedding model dimension {actual} does not match configured vector dimension {expected}"
            )

    async def embed_texts(self, texts: Sequence[str], model: str) -> list[list[float]]:
        self.validate_dimension(model)
        if not texts:
            return []
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        payload = {"input": list(texts), "model": model}
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        async with httpx.AsyncClient(base_url=self.settings.openai_base_url, timeout=60) as client:
            response = await client.post("/embeddings", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return [item["embedding"] for item in data["data"]]
