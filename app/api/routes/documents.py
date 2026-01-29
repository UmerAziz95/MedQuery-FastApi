import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.core.config import get_settings
from app.core.crash_logger import crash_logger
from app.db.session import AsyncSessionLocal, get_session
from app.models import Business, BusinessAdmin, Document, DocumentChunk, Workspace, WorkspaceConfig
from app.schemas.document import DocumentChunkOut, DocumentOut, DocumentStatusResponse, DocumentUploadResponse
from app.services.rag_ingest_service import RagIngestService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/businesses/{business_client_id}/workspaces/{workspace_id}",
    tags=["Documents"],
)
_INGEST_LOCK = asyncio.Lock()


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
    delay_before_start: bool = True,
) -> None:
    """Background task to process document ingestion with timeout and error handling."""
    import uuid
    import asyncio
    import signal
    
    # Small delay to ensure the response is sent first
    if delay_before_start:
        await asyncio.sleep(0.1)
    
    async with _INGEST_LOCK:
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
                    f"Processing started: document_id={document_id}, filename={document.filename}, "
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
                    logger.info(f"Processing completed successfully: document_id={document_id}")
                except asyncio.TimeoutError:
                    logger.error(f"Processing timed out for document {document_id}")
                    # Refresh document to get latest state
                    await bg_session.refresh(document)
                    document.status = "failed"
                    document.meta_json = "Processing timed out after 30 minutes"
                    await bg_session.commit()
                except MemoryError as mem_err:
                    # Log crash with detailed context
                    crash_logger.log_crash(
                        mem_err, type(mem_err), mem_err.__traceback__,
                        context={
                            "operation": "document_processing",
                            "document_id": document_id,
                            "document_filename": document.filename if document else None,
                            "business_client_id": business_client_id,
                            "workspace_id": workspace_id,
                        },
                        additional_info={
                            "chunk_words": chunk_words,
                            "overlap_words": overlap_words,
                        }
                    )
                    
                    logger.error(f"Out of memory error for document {document_id}: {mem_err}")
                    # Refresh document to get latest state
                    await bg_session.refresh(document)
                    document.status = "failed"
                    document.meta_json = (
                        f"Out of memory: PDF may be too large or complex. "
                        f"Try splitting the document or increasing container memory. "
                        f"Check logs/crashes/ for detailed crash report."
                    )
                    await bg_session.commit()
                except KeyboardInterrupt:
                    # Handle graceful shutdown
                    logger.warning(f"Processing interrupted for document {document_id}")
                    await bg_session.refresh(document)
                    document.status = "failed"
                    document.meta_json = "Processing interrupted"
                    await bg_session.commit()
                    raise
                except Exception:
                    # Re-raise to be caught by outer handler
                    raise
                
            except Exception as exc:
                # Determine if this is a critical crash
                is_critical = isinstance(exc, (MemoryError, SystemError, KeyboardInterrupt)) or \
                              "out of memory" in str(exc).lower() or \
                              "killed" in str(exc).lower()
                
                # Log with crash logger
                if is_critical:
                    crash_logger.log_crash(
                        exc, type(exc), exc.__traceback__,
                        context={
                            "operation": "document_processing",
                            "document_id": document_id,
                            "document_filename": document.filename if document else None,
                            "business_client_id": business_client_id,
                            "workspace_id": workspace_id,
                        },
                        additional_info={
                            "chunk_words": chunk_words,
                            "overlap_words": overlap_words,
                            "is_critical": True,
                        }
                    )
                else:
                    crash_logger.log_error(
                        exc, type(exc), exc.__traceback__,
                        context={
                            "operation": "document_processing",
                            "document_id": document_id,
                            "business_client_id": business_client_id,
                            "workspace_id": workspace_id,
                        },
                        severity="ERROR"
                    )
                
                logger.error(
                    f"Error in processing for document {document_id}: "
                    f"{type(exc).__name__}: {exc}",
                    exc_info=True
                )
                try:
                    # Try to update document status if we have it
                    if document:
                        # Refresh to get latest state
                        try:
                            await bg_session.refresh(document)
                        except Exception:
                            pass  # Document might have been deleted
                        
                        error_msg = str(exc)[:500]  # Limit error message length
                        document.status = "failed"
                        document.meta_json = (
                            f"{type(exc).__name__}: {error_msg}. "
                            f"Check logs/crashes/ for detailed {'crash' if is_critical else 'error'} report."
                        )
                        await bg_session.commit()
                        logger.info(f"Updated document {document_id} status to failed: {error_msg}")
                except Exception as commit_exc:
                    logger.error(f"Failed to update document status: {commit_exc}")
                    # Try one more time with a new session if possible
                    try:
                        async with AsyncSessionLocal() as recovery_session:
                            recovery_doc = (
                                await recovery_session.execute(
                                    select(Document).where(Document.id == doc_uuid)
                                )
                            ).scalar_one_or_none()
                            if recovery_doc:
                                recovery_doc.status = "failed"
                                recovery_doc.meta_json = (
                                    f"Processing failed: {type(exc).__name__}. "
                                    f"Check logs/crashes/ for details."
                                )
                                await recovery_session.commit()
                                logger.info(f"Recovered document {document_id} status via recovery session")
                    except Exception as recovery_exc:
                        logger.error(f"Failed to recover document status: {recovery_exc}")


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

        # Validate file size before storing (50MB limit)
        MAX_FILE_SIZE = 50 * 1024 * 1024
        service = RagIngestService()
        
        # Check file size during upload
        file_size = 0
        storage_dir = Path(service.settings.file_storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / file.filename
        
        # Stream and validate size simultaneously
        logger.debug(f"Streaming file content to disk: {file.filename}")
        with storage_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    # Delete partial file
                    storage_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"File too large: {file_size / (1024*1024):.1f}MB. "
                            f"Maximum allowed: {MAX_FILE_SIZE / (1024*1024):.0f}MB. "
                            f"Please split the document into smaller files."
                        )
                    )
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
        
        settings = get_settings()
        file_size_kb = file_size / 1024
        estimated_pages = max(1, int(file_size_kb / 50))
        
        if settings.rag_background_ingest:
            logger.info(
                f"File detected ({file_size_kb:.1f}KB, ~{estimated_pages} pages), "
                "using background processing for non-blocking upload"
            )
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
        else:
            logger.info(
                f"File detected ({file_size_kb:.1f}KB, ~{estimated_pages} pages), "
                "processing synchronously to avoid background resource contention"
            )
            await _process_document_background(
                str(document.id),
                business_client_id,
                workspace_id,
                chunk_words,
                overlap_words,
                delay_before_start=False,
            )
        
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


@router.post("/documents/{document_id}/reset")
async def reset_document(
    business_client_id: str,
    workspace_id: str,
    document_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> dict:
    """Reset a stuck processing document to allow retry."""
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
    
    if document.status == "processing":
        # Reset to processing to allow retry
        document.status = "processing"
        document.meta_json = "Reset for retry"
        await session.commit()
        logger.info(f"Document {document_id} reset for retry")
        return {"status": "reset", "message": "Document reset. You can retry processing."}
    else:
        return {"status": "unchanged", "message": f"Document status is {document.status}, no reset needed."}


@router.post("/documents/reset-stuck")
async def reset_stuck_documents(
    business_client_id: str,
    workspace_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> dict:
    """Reset all documents stuck in processing status (likely from crashes)."""
    from datetime import datetime, timedelta
    
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    
    # Find documents stuck in processing for more than 1 hour
    from sqlalchemy import or_
    cutoff_time = datetime.utcnow() - timedelta(hours=1)
    stmt = select(Document).where(
        Document.business_id == business.id,
        Document.workspace_id == workspace.id,
        Document.status == "processing"
    ).where(
        or_(
            Document.indexed_at == None,
            Document.indexed_at < cutoff_time
        )
    )
    stuck_documents = list((await session.execute(stmt)).scalars().all())
    
    reset_count = 0
    for doc in stuck_documents:
        # Check if document has any chunks - if not, it's truly stuck
        chunk_count = (
            await session.execute(
                select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
            )
        ).scalar() or 0
        
        if chunk_count == 0:
            doc.status = "failed"
            doc.meta_json = "Reset: Stuck in processing (likely crashed). Can be retried."
            reset_count += 1
    
    if reset_count > 0:
        await session.commit()
        logger.info(f"Reset {reset_count} stuck documents in workspace {workspace_id}")
    
    return {
        "status": "completed",
        "reset_count": reset_count,
        "message": f"Reset {reset_count} stuck document(s). They can now be retried."
    }
