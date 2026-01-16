from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
    business = (
        await session.execute(
            select(Business).where(Business.business_client_id == payload.business_client_id)
        )
    ).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not founds")
    workspace = (
        await session.execute(
            select(Workspace).where(
                Workspace.business_id == business.id,
                Workspace.workspace_id == payload.workspace_id,
            )
        )
    ).scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    config = (
        await session.execute(
            select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
        )
    ).scalar_one()

    service = ChatService()
    answer, sources, usage = await service.generate_response(
        session=session,
        business_id=business.id,
        workspace_id=workspace.id,
        user_id=payload.user_id,
        query=payload.query,
        prompt_engineering=payload.prompt_engineering,
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
