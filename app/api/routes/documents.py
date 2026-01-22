import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db.session import AsyncSessionLocal, get_session
from app.models import Business, BusinessAdmin, Document, DocumentChunk, Workspace, WorkspaceConfig
from app.schemas.document import DocumentChunkOut, DocumentOut, DocumentStatusResponse, DocumentUploadResponse
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
    """Background task to process document ingestion with timeout and error handling."""
    import uuid
    import asyncio
    import signal
    
    # Small delay to ensure the response is sent first
    await asyncio.sleep(0.1)
    
    async with AsyncSessionLocal() as bg_session:
        document = None
        try:
            logger.info(f"Background task started for document {document_id}")
            
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
                f"Background processing started: document_id={document_id}, filename={document.filename}, "
                f"chunk_words={chunk_words or config.chunk_words}, "
                f"overlap_words={overlap_words or config.overlap_words}"
            )
            
            # Add timeout to prevent hanging (30 minutes max)
            try:
                service = RagIngestService()
                await asyncio.wait_for(
                    service.ingest_document(bg_session, document, config, chunk_words, overlap_words),
                    timeout=1800.0  # 30 minutes
                )
                logger.info(f"Background processing completed successfully: document_id={document_id}")
            except asyncio.TimeoutError:
                logger.error(f"Background processing timed out for document {document_id}")
                document.status = "failed"
                document.meta_json = "Processing timed out after 30 minutes"
                await bg_session.commit()
            except MemoryError as mem_err:
                logger.error(f"Out of memory error for document {document_id}: {mem_err}")
                document.status = "failed"
                document.meta_json = f"Out of memory: PDF may be too large or complex. Try splitting the document or increasing container memory."
                await bg_session.commit()
            except Exception as proc_exc:
                # Re-raise to be caught by outer handler
                raise
            
        except Exception as exc:
            logger.error(
                f"Error in background processing for document {document_id}: "
                f"{type(exc).__name__}: {exc}",
                exc_info=True
            )
            try:
                # Try to update document status if we have it
                if document:
                    error_msg = str(exc)[:500]  # Limit error message length
                    document.status = "failed"
                    document.meta_json = f"{type(exc).__name__}: {error_msg}"
                    await bg_session.commit()
                    logger.info(f"Updated document {document_id} status to failed: {error_msg}")
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

        # Stream file to disk to avoid loading large files into memory
        logger.debug(f"Streaming file content to disk: {file.filename}")
        service = RagIngestService()
        storage_dir = Path(service.settings.file_storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / file.filename
        file_size = 0
        with storage_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                buffer.write(chunk)
        logger.info(f"File stored successfully: {file.filename}, size: {file_size} bytes")
        
        # Create document record with "processing" status
        document = Document(
            business_id=business.id,
            workspace_id=workspace.id,
            filename=file.filename,
            file_type=file_type,
            storage_path=str(storage_path),
            status="processing",
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)
        logger.info(f"Document record created: document_id={document.id}, status=processing")
        
        # Get config for processing decision
        config = (
            await session.execute(
                select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
            )
        ).scalar_one()
        
        # ALWAYS use background processing for better performance and non-blocking requests
        # Even small files can take time due to embedding API calls and database writes
        file_size_kb = file_size / 1024
        estimated_pages = max(1, int(file_size_kb / 50))
        
        logger.info(f"File detected ({file_size_kb:.1f}KB, ~{estimated_pages} pages), using background processing for non-blocking upload")
        try:
            background_tasks.add_task(
                _process_document_background,
                str(document.id),
                business_client_id,
                workspace_id,
                chunk_words,
                overlap_words,
            )
            logger.info(f"Background task added for document {document.id}")
        except Exception as task_error:
            logger.error(f"Failed to add background task: {task_error}", exc_info=True)
            # If background task fails, still return success but log the error
            # The document will remain in "processing" status and can be retried
            logger.warning(f"Background task setup failed, document {document.id} will remain in processing status")
        
        # Return immediately with processing status (non-blocking)
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
    documents = list((await session.execute(stmt)).scalars().all())
    
    # Add chunk count for each document
    result = []
    for doc in documents:
        chunk_count = (
            await session.execute(
                select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
            )
        ).scalar() or 0
        
        doc_dict = {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "status": doc.status,
            "chunk_count": chunk_count,
            "indexed_at": doc.indexed_at.isoformat() if doc.indexed_at else None,
            "meta_json": doc.meta_json,
        }
        result.append(DocumentOut(**doc_dict))
    
    return result


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document(
    business_client_id: str,
    workspace_id: str,
    document_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> DocumentStatusResponse:
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
    
    # Get chunk count
    chunk_count = (
        await session.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document.id)
        )
    ).scalar() or 0
    
    return DocumentStatusResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        status=document.status,
        chunk_count=chunk_count,
        indexed_at=document.indexed_at.isoformat() if document.indexed_at else None,
        meta_json=document.meta_json,
    )


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
