from typing import Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentChunk, WorkspaceConfig


def build_retrieval_query(
    business_id: uuid.UUID,
    workspace_id: uuid.UUID,
    query_embedding: Sequence[float],
    top_k: int,
    similarity_threshold: float,
):
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(
            DocumentChunk,
            Document.filename.label("filename"),
            distance.label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.business_id == business_id,
            DocumentChunk.workspace_id == workspace_id,
        )
        .order_by(distance)
        .limit(top_k)
    )
    if similarity_threshold > 0:
        stmt = stmt.where(distance <= 1 - similarity_threshold)
    return stmt


async def retrieve_chunks(
    session: AsyncSession,
    business_id: uuid.UUID,
    workspace_id: uuid.UUID,
    query_embedding: Sequence[float],
    top_k: int,
    similarity_threshold: float,
) -> list[tuple[DocumentChunk, str, float]]:
    stmt = build_retrieval_query(
        business_id=business_id,
        workspace_id=workspace_id,
        query_embedding=query_embedding,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    result = await session.execute(stmt)
    return [(row[0], row[1], float(row[2])) for row in result.all()]


async def get_workspace_config(session: AsyncSession, workspace_id: str) -> WorkspaceConfig:
    stmt = select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace_id)
    return (await session.execute(stmt)).scalar_one()
