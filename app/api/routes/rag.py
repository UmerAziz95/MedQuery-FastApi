from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models import Business, Workspace, WorkspaceConfig
from app.schemas.rag import RetrievalRequest, RetrievalResponse, RetrievedChunk
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import retrieve_chunks

router = APIRouter(prefix="/rag", tags=["RAG Retrieval"])


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    payload: RetrievalRequest,
    session: AsyncSession = Depends(get_session),
) -> RetrievalResponse:
    business = (
        await session.execute(
            select(Business).where(Business.business_client_id == payload.business_client_id)
        )
    ).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
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

    embedding_service = EmbeddingService()
    query_embedding = (await embedding_service.embed_texts([payload.query], config.embedding_model))[0]

    chunks = await retrieve_chunks(
        session=session,
        business_id=business.id,
        workspace_id=workspace.id,
        query_embedding=query_embedding,
        top_k=payload.top_k or config.top_k,
        similarity_threshold=config.similarity_threshold,
    )

    retrieved = [
        RetrievedChunk(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            filename=filename,
            page=chunk.page_number,
            score=1 - distance,
            content=chunk.content,
        )
        for chunk, filename, distance in chunks
    ]

    return RetrievalResponse(
        business_client_id=payload.business_client_id,
        workspace_id=payload.workspace_id,
        user_id=payload.user_id,
        query=payload.query,
        retrieved_chunks=retrieved,
    )
