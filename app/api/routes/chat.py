from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import NoResultFound

from app.core.chat_logger import log_chat
from app.db.session import get_session
from app.models import Business, Workspace, WorkspaceConfig
from app.schemas.rag import ChatRequest, ChatResponse, ChatUsage
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/generate", response_model=ChatResponse)
async def generate_chat(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    log_chat(
        "CHAT_REQUEST_RECEIVED",
        "Chat /generate request received",
        business_client_id=payload.business_client_id,
        workspace_id=payload.workspace_id,
        user_id=payload.user_id,
        query=payload.query[:200] if payload.query else "",
    )

    try:
        business = (
            await session.execute(
                select(Business).where(Business.business_client_id == payload.business_client_id)
            )
        ).scalar_one_or_none()
        if not business:
            log_chat("CHAT_ERROR", "Business not found", step="business_lookup", business_client_id=payload.business_client_id)
            raise HTTPException(status_code=404, detail="Business not found")
        log_chat("CHAT_BUSINESS_FOUND", "Business resolved", business_id=str(business.id))

        workspace = (
            await session.execute(
                select(Workspace).where(
                    Workspace.business_id == business.id,
                    Workspace.workspace_id == payload.workspace_id,
                )
            )
        ).scalar_one_or_none()
        if not workspace:
            log_chat("CHAT_ERROR", "Workspace not found", step="workspace_lookup", workspace_id=payload.workspace_id)
            raise HTTPException(status_code=404, detail="Workspace not found")
        log_chat("CHAT_WORKSPACE_FOUND", "Workspace resolved", workspace_id=str(workspace.id))

        config = (
            await session.execute(
                select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
            )
        ).scalar_one_or_none()
        if not config:
            log_chat("CHAT_ERROR", "Workspace config not found", step="config_lookup", workspace_id=str(workspace.id))
            raise HTTPException(status_code=404, detail="Workspace config not found. Create or seed config for this workspace.")
        config_prompt = (getattr(config, "prompt_engineering", None) or "").strip()
        log_chat(
            "CHAT_CONFIG_FOUND",
            "Workspace config loaded",
            embedding_model=config.embedding_model,
            chat_model=config.chat_model_default,
            prompt_from_db_len=len(config_prompt),
        )

        # Workspace config (DB) is primary; payload is override only when config is empty
        prompt_engineering = (
            config_prompt
            or (payload.prompt_engineering or "").strip()
            or "You are a medical assistant. Provide concise answers based on the context."
        )

        service = ChatService()
        answer, sources, usage = await service.generate_response(
            session=session,
            business_id=business.id,
            workspace_id=workspace.id,
            user_id=payload.user_id,
            query=payload.query,
            prompt_engineering=prompt_engineering,
            config=config,
            override=payload.chat_config_override.model_dump() if payload.chat_config_override else None,
        )

        return ChatResponse(
            business_client_id=payload.business_client_id,
            workspace_id=payload.workspace_id,
            user_id=payload.user_id,
            query=payload.query,
            answer=answer,
            sources=sources,
            usage=ChatUsage(
                model=usage.get("model", config.chat_model_default),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )
    except HTTPException:
        raise
    except NoResultFound as e:
        log_chat("CHAT_ERROR", "NoResultFound in chat route", step="lookup", error=str(e), error_type="NoResultFound")
        raise HTTPException(status_code=404, detail="Resource not found (business, workspace, or config).")
    except Exception as e:
        log_chat("CHAT_ERROR", "Chat generate failed", step="generate", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
