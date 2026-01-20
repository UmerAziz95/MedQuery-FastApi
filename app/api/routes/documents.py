import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db.session import AsyncSessionLocal
from app.models import Business, BusinessAdmin, Document, DocumentChunk, Workspace, WorkspaceConfig
from app.schemas.document import DocumentChunkOut, DocumentOut, DocumentUploadResponse
from app.services.rag_ingest_service import RagIngestService

logger = logging.getLogger(__name__)

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


async def _process_document_background(
    document_id: str,
    business_client_id: str,
    workspace_id: str,
    chunk_words: int | None,
    overlap_words: int | None,
) -> None:
    """Background task to process document ingestion."""
    import uuid
    async with AsyncSessionLocal() as bg_session:
        try:
            # Convert string ID to UUID
            doc_uuid = uuid.UUID(document_id)
            
            # Get document
            document = (
                await bg_session.execute(
                    select(Document).where(Document.id == doc_uuid)
                )
            ).scalar_one_or_none()
            
            if not document:
                logger.error(f"Document {document_id} not found for background processing")
                return
            
            # Get workspace config
            workspace = (
                await bg_session.execute(
                    select(Workspace).where(Workspace.workspace_id == workspace_id)
                )
            ).scalar_one_or_none()
            
            if not workspace:
                logger.error(f"Workspace {workspace_id} not found")
                document.status = "failed"
                document.meta_json = "Workspace not found"
                await bg_session.commit()
                return
            
            config = (
                await bg_session.execute(
                    select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
                )
            ).scalar_one()
            
            logger.info(
                f"Background processing started: document_id={document_id}, "
                f"chunk_words={chunk_words or config.chunk_words}, "
                f"overlap_words={overlap_words or config.overlap_words}"
            )
            
            service = RagIngestService()
            await service.ingest_document(bg_session, document, config, chunk_words, overlap_words)
            logger.info(f"Background processing completed: document_id={document_id}")
            
        except Exception as exc:
            logger.error(
                f"Error in background processing for document {document_id}: "
                f"{type(exc).__name__}: {exc}",
                exc_info=True
            )
            try:
                # Try to update document status if we have it
                if 'document' in locals() and document:
                    document.status = "failed"
                    document.meta_json = f"{type(exc).__name__}: {str(exc)}"
                    await bg_session.commit()
            except Exception as commit_exc:
                logger.error(f"Failed to update document status: {commit_exc}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    business_client_id: str,
    workspace_id: str,
    file: UploadFile = File(...),
    chunk_words: int | None = None,
    overlap_words: int | None = None,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> DocumentUploadResponse:
    """Upload document and process in background for fast response."""
    logger.info(
        f"Document upload request: business={business_client_id}, "
        f"workspace={workspace_id}, filename={file.filename}, "
        f"size={file.size if hasattr(file, 'size') else 'unknown'}"
    )
    
    try:
        business, workspace = await _get_workspace(session, business_client_id, workspace_id)
        _ensure_access(admin, business)
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        file_type = file.filename.split(".")[-1].lower()
        if file_type not in {"pdf", "txt"}:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and TXT files are supported.")

        # Read and save file
        logger.debug(f"Reading file content: {file.filename}")
        service = RagIngestService()
        data = await file.read()
        file_size = len(data)
        logger.info(f"File read successfully: {file.filename}, size: {file_size} bytes")
        
        storage_path = service.store_upload(file.filename, data)
        logger.debug(f"File stored at: {storage_path}")
        
        # Create document record with "processing" status
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
        logger.info(f"Document record created: document_id={document.id}, status=processing")
        
        # Add background task to process document
        background_tasks.add_task(
            _process_document_background,
            str(document.id),
            business_client_id,
            workspace_id,
            chunk_words,
            overlap_words,
        )
        logger.info(f"Background task added for document {document.id}")
        
        # Return immediately with processing status
        return DocumentUploadResponse(
            document_id=str(document.id),
            status="processing",
            business_client_id=business_client_id,
            workspace_id=workspace_id,
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"Unexpected error in upload_document endpoint: {type(exc).__name__}: {exc}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {type(exc).__name__}: {str(exc)}"
        ) from exc


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
