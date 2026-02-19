import json
import uuid

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.chat_logger import log_chat
from app.core.config import get_settings
from app.models import ChatRequest, ChatResponse, WorkspaceConfig
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import retrieve_chunks
from app.services.system_config_service import get_openai_api_key

PROMPT_PREVIEW_MAX = 800
ANSWER_PREVIEW_MAX = 500


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
        log_chat(
            "CHAT_QUERY_RECEIVED",
            "Query received in system",
            business_id=str(business_id),
            workspace_id=str(workspace_id),
            user_id=user_id,
            query=query,
            query_word_count=len(query.split()),
        )

        openai_api_key = (await get_openai_api_key(session)) or self.settings.openai_api_key

        try:
            log_chat("CHAT_EMBEDDING_START", "Creating query embedding", embedding_model=config.embedding_model, use_local=config.use_local_embeddings)
            embedding = (await self.embedding_service.embed_texts(
                [query],
                config.embedding_model,
                use_local=config.use_local_embeddings,
                openai_api_key=openai_api_key,
            ))[0]
            log_chat(
                "CHAT_EMBEDDING_DONE",
                "Query vector created",
                vector_dim=len(embedding),
                vector_preview=embedding[:8],
            )
        except Exception as e:
            log_chat("CHAT_ERROR", f"Step failed: embedding", step="embedding", error=str(e), error_type=type(e).__name__)
            raise

        try:
            log_chat("CHAT_RETRIEVAL_START", "Searching RAG for matching chunks", top_k=config.top_k, similarity_threshold=config.similarity_threshold)
            chunks = await retrieve_chunks(
                session=session,
                business_id=business_id,
                workspace_id=workspace_id,
                query_embedding=embedding,
                top_k=config.top_k,
                similarity_threshold=config.similarity_threshold,
            )
            chunk_ids = [str(c.id) for c, _, _ in chunks]
            chunk_summaries = [{"chunk_id": str(c.id), "filename": fn, "page": c.page_number, "distance": float(dist)} for c, fn, dist in chunks]
            log_chat(
                "CHAT_RETRIEVAL_DONE",
                f"Matched {len(chunks)} chunks",
                chunk_count=len(chunks),
                chunk_ids=chunk_ids,
                chunks_matched=chunk_summaries,
            )
        except Exception as e:
            log_chat("CHAT_ERROR", f"Step failed: retrieval", step="retrieval", error=str(e), error_type=type(e).__name__)
            raise

        context = "\n\n".join([chunk.content for chunk, _, _ in chunks])
        system_prompt = prompt_engineering
        user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        full_prompt_preview = f"[SYSTEM]\n{system_prompt[:PROMPT_PREVIEW_MAX]}{'...' if len(system_prompt) > PROMPT_PREVIEW_MAX else ''}\n[USER]\n{user_content[:PROMPT_PREVIEW_MAX]}{'...' if len(user_content) > PROMPT_PREVIEW_MAX else ''}"
        log_chat(
            "CHAT_PROMPT_BUILT",
            "Final prompt assembled for ChatGPT API",
            system_prompt_len=len(system_prompt),
            context_len=len(context),
            user_content_len=len(user_content),
            prompt_preview=full_prompt_preview,
        )

        model = (override.get("model") or config.chat_model_default) if override else config.chat_model_default
        model = model or self.settings.default_chat_model
        temperature = (
            (override.get("temperature") if override and "temperature" in override else config.chat_temperature_default)
            if override
            else config.chat_temperature_default
        )
        if temperature is None:
            temperature = config.chat_temperature_default
        max_tokens = (
            (override.get("max_tokens") if override and "max_tokens" in override else config.chat_max_tokens_default)
            if override
            else config.chat_max_tokens_default
        )
        if max_tokens is None:
            max_tokens = config.chat_max_tokens_default

        try:
            request_content_len = sum(len(m.get("content", "")) for m in messages)
            log_chat("CHAT_OPENAI_CALL", f"Calling ChatGPT API (model={model})", model=model, temperature=temperature, max_tokens=max_tokens, message_count=len(messages), request_content_len=request_content_len)
            response = await self._call_openai(messages, model, temperature, max_tokens, openai_api_key=openai_api_key)
        except Exception as e:
            log_chat("CHAT_ERROR", "Step failed: OpenAI API call", step="openai_call", error=str(e), error_type=type(e).__name__)
            raise

        answer = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        log_chat(
            "CHAT_OPENAI_RESPONSE",
            "GPT response received",
            answer_preview=answer[:ANSWER_PREVIEW_MAX] + ("..." if len(answer) > ANSWER_PREVIEW_MAX else ""),
            full_answer=answer,
            answer_len=len(answer),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw_response=json.dumps(response, ensure_ascii=False),
        )

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

        try:
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
            log_chat("CHAT_SAVED", "Chat request and response saved to DB", request_id=str(chat_request.id))
        except Exception as e:
            log_chat("CHAT_ERROR", f"Step failed: save to DB", step="save", error=str(e), error_type=type(e).__name__)
            raise

        return answer, sources, usage

    def _mask_api_key(self, key: str | None) -> str:
        """Return masked key for logging (first 7 + ... + last 4). Never log full key."""
        if not key or not isinstance(key, str):
            return "(empty or not set)"
        key = key.strip()
        if len(key) <= 11:
            return "(too short to mask)"
        return f"{key[:7]}...{key[-4:]}"

    async def _call_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        openai_api_key: str | None = None,
    ) -> dict:
        key = (openai_api_key or self.settings.openai_api_key) or ""
        key_set = bool(key and key.strip())
        log_chat(
            "CHAT_OPENAI_KEY_CHECK",
            "OpenAI API key from DB/env (masked)",
            key_set=key_set,
            key_length=len(key.strip()) if key else 0,
            masked_key=self._mask_api_key(key),
        )
        if not key_set:
            raise RuntimeError("OPENAI_API_KEY not configured. Set it in System configurations.")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request_body_log = json.dumps({"model": model, "message_count": len(messages), "temperature": temperature, "max_tokens": max_tokens, "message_content_lens": [len(m.get("content", "")) for m in messages]}, ensure_ascii=False)
        headers = {"Authorization": f"Bearer {key}"}
        try:
            async with httpx.AsyncClient(base_url=self.settings.openai_base_url, timeout=60) as client:
                log_chat("CHAT_OPENAI_SEND", "Sending request to OpenAI", model=model, base_url=self.settings.openai_base_url, request_body=request_body_log)
                response = await client.post("/chat/completions", json=payload, headers=headers)
                raw_response_text = response.text or ""
                status_code = response.status_code
                # Always log the raw API response first (success or error) so we never miss it
                log_chat(
                    "CHAT_OPENAI_API_RESPONSE",
                    "OpenAI API raw response (before raise_for_status)",
                    status_code=status_code,
                    response_body=raw_response_text,
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    log_chat("CHAT_ERROR", "OpenAI API returned error", step="openai_call", status_code=status_code, response_body=raw_response_text, error=str(e))
                    raise
                try:
                    response_json = response.json()
                except Exception as parse_err:
                    log_chat("CHAT_ERROR", "OpenAI response JSON parse failed", step="openai_call", response_body=raw_response_text, error=str(parse_err), error_type=type(parse_err).__name__)
                    raise
                log_chat("CHAT_OPENAI_RAW_RESPONSE", "OpenAI API raw response body (success)", raw_response=json.dumps(response_json, ensure_ascii=False))
                return response_json
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            log_chat("CHAT_ERROR", "OpenAI request failed (network/timeout/other)", step="openai_call", error=str(e), error_type=type(e).__name__)
            raise
