import json
import uuid

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import ChatRequest, ChatResponse, WorkspaceConfig
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import retrieve_chunks


class ChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()

    async def generate_response(
        self,
        session: AsyncSession,
        business_id: uuid.UUID,
        workspace_id: uuid.UUID,
        user_id: str,
        query: str,
        prompt_engineering: str,
        config: WorkspaceConfig,
        override: dict | None = None,
    ) -> tuple[str, list[dict], dict]:
        embedding = (await self.embedding_service.embed_texts([query], config.embedding_model, use_local=config.use_local_embeddings))[0]
        chunks = await retrieve_chunks(
            session=session,
            business_id=business_id,
            workspace_id=workspace_id,
            query_embedding=embedding,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
        )
        context = "\n\n".join([chunk.content for chunk, _, _ in chunks])
        system_prompt = prompt_engineering
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}",
            },
        ]
        model = override.get("model", config.chat_model_default) if override else config.chat_model_default
        temperature = (
            override.get("temperature", config.chat_temperature_default)
            if override
            else config.chat_temperature_default
        )
        max_tokens = (
            override.get("max_tokens", config.chat_max_tokens_default)
            if override
            else config.chat_max_tokens_default
        )

        response = await self._call_openai(messages, model, temperature, max_tokens)
        answer = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})

        sources = [
            {
                "document_id": str(chunk.document_id),
                "filename": filename,
                "page": chunk.page_number,
                "chunk_id": str(chunk.id),
                "snippet": chunk.content[:240],
            }
            for chunk, filename, _ in chunks
        ]

        chat_request = ChatRequest(
            business_id=business_id,
            workspace_id=workspace_id,
            user_id=user_id,
            query_text=query,
            retrieved_chunk_ids=json.dumps([str(chunk.id) for chunk, _, _ in chunks]),
        )
        session.add(chat_request)
        await session.flush()
        chat_response = ChatResponse(
            request_id=chat_request.id,
            answer_text=answer,
            sources_json=json.dumps(sources),
            model_used=model,
            tokens_json=json.dumps(usage),
        )
        session.add(chat_response)
        await session.commit()

        return answer, sources, usage

    async def _call_openai(self, messages: list[dict], model: str, temperature: float, max_tokens: int) -> dict:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        async with httpx.AsyncClient(base_url=self.settings.openai_base_url, timeout=60) as client:
            response = await client.post("/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
