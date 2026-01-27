import gc
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    from pypdf import PdfReader
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.crash_logger import crash_logger
from app.models import Document, DocumentChunk, WorkspaceConfig
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024
# Ultra-conservative batch size for embeddings
ULTRA_SMALL_BATCH = 10
# Memory threshold: abort if over 2GB (very conservative for 4GB limit)
MEMORY_THRESHOLD_GB = 2.0


class RagIngestService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
    
    def _check_memory(self) -> tuple[float, float]:
        """Check current memory usage. Returns (used_gb, percent)."""
        if not HAS_PSUTIL:
            return 0.0, 0.0
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            used_gb = mem_info.rss / (1024 ** 3)  # Convert to GB
            percent = process.memory_percent()
            return used_gb, percent
        except Exception:
            return 0.0, 0.0
    
    def _check_memory_safe(self) -> bool:
        """Check if memory usage is safe (< 2GB used)."""
        if not HAS_PSUTIL:
            return True
        used_gb, percent = self._check_memory()
        used_mb = used_gb * 1024
        threshold_mb = MEMORY_THRESHOLD_GB * 1024
        
        if used_gb > MEMORY_THRESHOLD_GB:
            # Log memory pressure
            crash_logger.log_memory_pressure(
                current_memory_mb=used_mb,
                threshold_mb=threshold_mb,
                context={
                    "operation": "document_ingestion",
                    "memory_used_gb": used_gb,
                    "memory_percent": percent,
                }
            )
            logger.warning(f"Memory threshold exceeded: {used_gb:.2f}GB ({percent:.1f}%)")
            return False
        return True
    
    def _validate_file_size(self, file_path: Path) -> None:
        """Validate file size before processing."""
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size / (1024*1024):.1f}MB. "
                f"Maximum allowed: {MAX_FILE_SIZE / (1024*1024):.0f}MB. "
                f"Please split the document into smaller files."
            )
        logger.info(f"File size validated: {file_size / 1024:.1f}KB")

    async def ingest_document(
        self,
        session: AsyncSession,
        document: Document,
        config: WorkspaceConfig,
        chunk_words_override: int | None = None,
        overlap_words_override: int | None = None,
    ) -> None:
        total_start = time.time()
        logger.info(
            f"Starting document ingestion: document_id={document.id}, "
            f"filename={document.filename}, file_type={document.file_type}"
        )

        try:
            file_path = Path(document.storage_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {document.storage_path}")

            # Validate file size upfront
            self._validate_file_size(file_path)

            # Check initial memory
            if not self._check_memory_safe():
                raise MemoryError("Memory usage too high at start of processing")

            chunk_words = chunk_words_override or config.chunk_words
            overlap_words = overlap_words_override or config.overlap_words

            chunk_count = 0
            chunk_index = 0
            page_count = 0
            embedding_batches = 0
            last_progress_update = time.time()

            async def update_progress_status(force: bool = False) -> None:
                """Update document status with progress information."""
                nonlocal last_progress_update
                current_time = time.time()
                if force or (current_time - last_progress_update) >= 30:
                    try:
                        await session.refresh(document)
                        progress_msg = (
                            f"Processing: {page_count} pages, {chunk_count} chunks, "
                            f"{embedding_batches} batches"
                        )
                        document.meta_json = progress_msg
                        await session.commit()
                        last_progress_update = current_time
                        logger.info(f"Progress: {progress_msg}")
                    except Exception as update_err:
                        logger.warning(f"Failed to update progress: {update_err}")

            async def process_small_batch(batch_chunks: list[tuple[int | None, str]]) -> None:
                """Process a very small batch of chunks (ultra-conservative)."""
                nonlocal chunk_index, chunk_count, embedding_batches

                if not batch_chunks:
                    return

                # Strict memory check before processing
                if not self._check_memory_safe():
                    raise MemoryError(
                        f"Memory threshold ({MEMORY_THRESHOLD_GB}GB) exceeded. "
                        f"Stopping processing to prevent crash."
                    )

                batch_texts = [chunk for _, chunk in batch_chunks]
                embedding_batches += 1
                
                try:
                    # Process embeddings with small batch
                    batch_embeddings = await self.embedding_service.embed_texts(
                        batch_texts, config.embedding_model, batch_size=len(batch_texts)
                    )

                    if len(batch_embeddings) != len(batch_chunks):
                        raise RuntimeError(
                            f"Embedding count mismatch: {len(batch_embeddings)} vs {len(batch_chunks)}"
                        )

                    # Create chunk objects
                    chunk_objects = [
                        DocumentChunk(
                            business_id=document.business_id,
                            workspace_id=document.workspace_id,
                            document_id=document.id,
                            chunk_index=chunk_index + index,
                            page_number=page,
                            content=content,
                            embedding=embedding,
                        )
                        for index, ((page, content), embedding) in enumerate(zip(batch_chunks, batch_embeddings))
                    ]

                    # Commit immediately
                    try:
                        session.add_all(chunk_objects)
                        await session.commit()
                    except Exception:
                        await session.rollback()
                        raise

                    chunk_count += len(batch_chunks)
                    chunk_index += len(batch_chunks)

                    # Aggressive cleanup
                    del batch_texts, batch_embeddings, chunk_objects, batch_chunks
                    gc.collect()
                    
                    # Update progress
                    await update_progress_status()
                    
                except MemoryError:
                    # Re-raise memory errors
                    raise
                except Exception as e:
                    logger.error(f"Error processing batch: {e}", exc_info=True)
                    raise

            # Process document based on type
            try:
                if document.file_type == "pdf":
                    # Process PDF pages one at a time with immediate commits
                    for page_number, page_chunks in self._iter_pages_ultra_safe(
                        file_path, chunk_words, overlap_words
                    ):
                        page_count = max(page_count, page_number)
                        
                        # Process chunks from this page in very small batches
                        pending_chunks: list[tuple[int | None, str]] = []
                        for chunk in page_chunks:
                            pending_chunks.append((page_number, chunk))
                            
                            # Process immediately when we have a small batch
                            if len(pending_chunks) >= ULTRA_SMALL_BATCH:
                                await process_small_batch(pending_chunks)
                                pending_chunks = []
                        
                        # Process remaining chunks from this page
                        if pending_chunks:
                            await process_small_batch(pending_chunks)
                            pending_chunks = []
                        
                        # Force GC after each page
                        gc.collect()
                        
                        # Check memory after each page
                        if not self._check_memory_safe():
                            raise MemoryError(
                                f"Memory threshold exceeded after processing {page_count} pages. "
                                f"Stopping to prevent crash."
                            )
                        
                        # Update progress every 5 pages
                        if page_count % 5 == 0:
                            await update_progress_status(force=True)
                            used_gb, percent = self._check_memory()
                            logger.info(
                                f"Page {page_count}: {chunk_count} chunks, "
                                f"Memory: {used_gb:.2f}GB ({percent:.1f}%)"
                            )
                else:
                    # Process text files in small chunks
                    text = file_path.read_text(encoding="utf-8")
                    all_chunks = self._chunk_text_streaming(text, chunk_words, overlap_words)
                    del text
                    
                    pending_chunks: list[tuple[int | None, str]] = []
                    for chunk in all_chunks:
                        pending_chunks.append((None, chunk))
                        
                        if len(pending_chunks) >= ULTRA_SMALL_BATCH:
                            await process_small_batch(pending_chunks)
                            pending_chunks = []
                    
                    if pending_chunks:
                        await process_small_batch(pending_chunks)
                    
            except MemoryError as mem_error:
                # Log crash with detailed context
                crash_logger.log_crash(
                    mem_error, type(mem_error), mem_error.__traceback__,
                    context={
                        "operation": "document_ingestion",
                        "document_id": str(document.id),
                        "document_filename": document.filename,
                        "file_type": document.file_type,
                        "pages_processed": page_count,
                        "chunks_processed": chunk_count,
                        "embedding_batches": embedding_batches,
                    },
                    additional_info={
                        "chunk_words": chunk_words,
                        "overlap_words": overlap_words,
                        "file_path": str(file_path),
                        "file_size_bytes": file_path.stat().st_size if file_path.exists() else None,
                    }
                )
                
                logger.error(f"Out of memory while processing document: {mem_error}")
                await session.refresh(document)
                document.status = "failed"
                document.meta_json = (
                    f"Memory error: {str(mem_error)}. "
                    f"Processed {page_count} pages, {chunk_count} chunks before failure. "
                    f"Try splitting the document into smaller files. "
                    f"Check logs/crashes/ for detailed crash report."
                )
                await session.commit()
                raise
            except Exception as process_error:
                # Log error with context
                crash_logger.log_error(
                    process_error, type(process_error), process_error.__traceback__,
                    context={
                        "operation": "document_ingestion",
                        "document_id": str(document.id),
                        "document_filename": document.filename,
                        "file_type": document.file_type,
                        "pages_processed": page_count,
                        "chunks_processed": chunk_count,
                    },
                    severity="ERROR"
                )
                
                logger.error(f"Error processing document: {process_error}", exc_info=True)
                await session.refresh(document)
                document.status = "failed"
                document.meta_json = (
                    f"Processing error: {type(process_error).__name__}: {str(process_error)}. "
                    f"Check logs/crashes/ for detailed error report."
                )
                await session.commit()
                raise

            # Final status update
            extraction_time = time.time() - total_start
            logger.info(
                f"Document processed: {page_count} pages, {chunk_count} chunks created "
                f"in {extraction_time:.2f}s ({embedding_batches} embedding batches)"
            )

            if chunk_count == 0:
                logger.warning(f"Document {document.id} produced no chunks - marking as empty")
                await session.refresh(document)
                document.status = "empty"
                document.indexed_at = datetime.utcnow()
                await session.commit()
                return

            await session.refresh(document)
            document.status = "indexed"
            document.indexed_at = datetime.utcnow()
            document.meta_json = f"Successfully indexed: {chunk_count} chunks from {page_count} pages"
            await session.commit()

            total_time = time.time() - total_start
            logger.info(
                f"Document {document.id} successfully indexed with {chunk_count} chunks. "
                f"Total time: {total_time:.2f}s"
            )
                
        except Exception as e:
            # Log crash with full context
            is_critical = isinstance(e, (MemoryError, SystemError, KeyboardInterrupt))
            
            crash_logger.log_crash(
                e, type(e), e.__traceback__,
                context={
                    "operation": "document_ingestion",
                    "document_id": str(document.id),
                    "document_filename": document.filename,
                    "file_type": document.file_type,
                },
                additional_info={
                    "file_path": str(file_path) if 'file_path' in locals() else None,
                    "is_critical": is_critical,
                }
            ) if is_critical else crash_logger.log_error(
                e, type(e), e.__traceback__,
                context={
                    "operation": "document_ingestion",
                    "document_id": str(document.id),
                    "document_filename": document.filename,
                },
                severity="ERROR"
            )
            
            logger.error(
                f"Error ingesting document {document.id} ({document.filename}): {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            try:
                await session.refresh(document)
                if document.status == "processing":
                    document.status = "failed"
                    document.meta_json = (
                        f"{type(e).__name__}: {str(e)[:500]}. "
                        f"Check logs/crashes/ for detailed crash report."
                    )
                    await session.commit()
            except Exception as status_error:
                logger.error(f"Failed to update document status: {status_error}")
            raise

    async def reindex_document(self, session: AsyncSession, document: Document, config: WorkspaceConfig) -> None:
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        document.status = "processing"
        await session.commit()
        await self.ingest_document(session, document, config)

    def _iter_pages_ultra_safe(
        self, file_path: Path, chunk_words: int, overlap_words: int
    ) -> Iterable[tuple[int, list[str]]]:
        """Process PDF pages with maximum memory safety."""
        doc = None
        try:
            if HAS_PYMUPDF:
                logger.debug("Using PyMuPDF for ultra-safe PDF processing")
                # Open PDF
                doc = fitz.open(str(file_path))
                total_pages = len(doc)
                logger.info(f"Processing PDF with {total_pages} pages (ultra-safe mode)")
                
                for index in range(total_pages):
                    page_number = index + 1
                    page = None
                    try:
                        # Extract text from single page
                        page = doc[index]
                        text = page.get_text() or ""
                        
                        # Immediately delete page reference
                        del page
                        page = None
                        
                        # Chunk text immediately
                        page_chunks = self._chunk_text_streaming(text, chunk_words, overlap_words)
                        
                        # Delete text immediately
                        del text
                        
                        # Yield chunks
                        yield page_number, page_chunks
                        
                        # Delete chunks from memory
                        del page_chunks
                        
                        # Aggressive GC every 2 pages
                        if (index + 1) % 2 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_number}: {e}")
                        if page:
                            del page
                        yield page_number, []
                        continue
            else:
                logger.debug("Using pypdf for ultra-safe PDF processing")
                reader = PdfReader(str(file_path), strict=False)
                total_pages = len(reader.pages)
                logger.info(f"Processing PDF with {total_pages} pages (ultra-safe mode)")
                
                for index, page in enumerate(reader.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                        page_chunks = self._chunk_text_streaming(text, chunk_words, overlap_words)
                        del page, text
                        yield index, page_chunks
                        del page_chunks
                        
                        if index % 2 == 0:
                            gc.collect()
                    except Exception as e:
                        logger.warning(f"Error extracting page {index}: {e}")
                        yield index, []
                        continue
        finally:
            if doc:
                try:
                    doc.close()
                    del doc
                except Exception:
                    pass
            # Triple GC for maximum cleanup
            gc.collect()
            gc.collect()
            gc.collect()
    
    def _iter_chunks(
        self, file_path: Path, file_type: str, chunk_words: int, overlap_words: int
    ) -> Iterable[tuple[int | None, str]]:
        """Legacy iterator for non-PDF files."""
        if file_type == "pdf":
            for page_number, page_chunks in self._iter_pages_ultra_safe(file_path, chunk_words, overlap_words):
                for chunk in page_chunks:
                    yield page_number, chunk
        else:
            text = file_path.read_text(encoding="utf-8")
            for chunk in self._chunk_text_streaming(text, chunk_words, overlap_words):
                yield None, chunk

    def _chunk_text_streaming(self, text: str, chunk_words: int, overlap_words: int) -> list[str]:
        """Chunk text efficiently."""
        if overlap_words >= chunk_words:
            overlap_words = 0
        
        words = text.split()
        if not words:
            return []
        
        chunks = []
        start = 0
        total_words = len(words)
        
        while start < total_words:
            end = min(total_words, start + chunk_words)
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            
            start = end - overlap_words
            if start < 0:
                start = 0
            if start >= total_words:
                break
        
        # Clear words list
        del words
        return chunks
    
    def _chunk_text(self, text: str, chunk_words: int, overlap_words: int) -> list[str]:
        """Legacy method."""
        return self._chunk_text_streaming(text, chunk_words, overlap_words)
