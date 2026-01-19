from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db.session import get_session
from app.models import Business, BusinessAdmin, Document, DocumentChunk, Workspace, WorkspaceConfig
from app.schemas.document import DocumentChunkOut, DocumentOut, DocumentUploadResponse
from app.services.rag_ingest_service import RagIngestService

router = APIRouter(
    prefix="/admin/businesses/{business_client_id}/workspaces/{workspace_id}",
    tags=["Documents"],
)


async def _get_workspace(
    session: AsyncSession, business_client_id: str, workspace_id: str
) -> tuple[Business, Workspace]:
    business = (
        await session.execute(
            select(Business).where(Business.business_client_id == business_client_id)
        )
    ).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    workspace = (
        await session.execute(
            select(Workspace).where(
                Workspace.business_id == business.id,
                Workspace.workspace_id == workspace_id,
            )
        )
    ).scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return business, workspace


def _ensure_access(admin: BusinessAdmin, business: Business) -> None:
    if admin.role != "super_admin" and admin.business_id != business.id:
        raise HTTPException(status_code=403, detail="Not allowed")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    business_client_id: str,
    workspace_id: str,
    file: UploadFile = File(...),
    chunk_words: int | None = None,
    overlap_words: int | None = None,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> DocumentUploadResponse:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_type = file.filename.split(".")[-1].lower()
    if file_type not in {"pdf", "txt"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    service = RagIngestService()
    data = await file.read()
    storage_path = service.store_upload(file.filename, data)
    document = Document(
        business_id=business.id,
        workspace_id=workspace.id,
        filename=file.filename,
        file_type=file_type,
        storage_path=storage_path,
        status="processing",
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    config = (
        await session.execute(
            select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
        )
    ).scalar_one()
    try:
        await service.ingest_document(session, document, config, chunk_words, overlap_words)
    except HTTPException:
        raise
    except ValueError as exc:
        document.status = "failed"
        document.meta_json = str(exc)
        await session.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        document.status = "failed"
        document.meta_json = str(exc)
        await session.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        document.status = "failed"
        document.meta_json = str(exc)
        await session.commit()
        raise HTTPException(status_code=500, detail="Document ingest failed") from exc

    return DocumentUploadResponse(
        document_id=str(document.id),
        status=document.status,
        business_client_id=business_client_id,
        workspace_id=workspace_id,
    )


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    business_client_id: str,
    workspace_id: str,
    document_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> dict:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    stmt = select(Document).where(
        Document.id == document_id,
        Document.business_id == business.id,
        Document.workspace_id == workspace.id,
    )
    document = (await session.execute(stmt)).scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    config = (
        await session.execute(
            select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
        )
    ).scalar_one()
    await RagIngestService().reindex_document(session, document, config)
    return {"status": "reindexed"}


@router.post("/reindex-all")
async def reindex_all(
    business_client_id: str,
    workspace_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> dict:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    documents = (
        await session.execute(
            select(Document).where(
                Document.business_id == business.id, Document.workspace_id == workspace.id
            )
        )
    ).scalars()
    config = (
        await session.execute(
            select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
        )
    ).scalar_one()
    service = RagIngestService()
    for document in documents:
        await service.reindex_document(session, document, config)
    return {"status": "reindexed_all"}


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    business_client_id: str,
    workspace_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> list[DocumentOut]:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    stmt = select(Document).where(
        Document.business_id == business.id, Document.workspace_id == workspace.id
    )
    return list((await session.execute(stmt)).scalars().all())


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    business_client_id: str,
    workspace_id: str,
    document_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> DocumentOut:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    stmt = select(Document).where(
        Document.id == document_id,
        Document.business_id == business.id,
        Document.workspace_id == workspace.id,
    )
    document = (await session.execute(stmt)).scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/documents/{document_id}")
async def delete_document(
    business_client_id: str,
    workspace_id: str,
    document_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> dict:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    stmt = select(Document).where(
        Document.id == document_id,
        Document.business_id == business.id,
        Document.workspace_id == workspace.id,
    )
    document = (await session.execute(stmt)).scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.delete(document)
    await session.commit()
    return {"status": "deleted"}


@router.get("/documents/{document_id}/chunks", response_model=list[DocumentChunkOut])
async def list_document_chunks(
    business_client_id: str,
    workspace_id: str,
    document_id: str,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> list[DocumentChunkOut]:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    stmt = (
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.business_id == business.id,
            DocumentChunk.workspace_id == workspace.id,
        )
        .offset(offset)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())
