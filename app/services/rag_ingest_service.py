"""
RAG Ingestion Service - Optimized for Batch Operations

Following ChromaDB pattern:
1. Extract all text (synchronous, fast)
2. Create all chunks (synchronous, instant)
3. Generate ALL embeddings in batch (single API call)
4. Insert ALL chunks in single database transaction (fast)
"""
import gc
import logging
import multiprocessing
import sys
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

# pypdf is used for memory-bounded PDF parsing (even if PyMuPDF is installed)
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.crash_logger import crash_logger
from app.models import Document, DocumentChunk, WorkspaceConfig
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024
# Stay under 2.5GB so container (4GB limit) never gets OOM-killed
MEMORY_THRESHOLD_GB = 2.5
# Very small batches to keep memory low (8 chunks = ~few MB per batch)
INGEST_BATCH_CHUNKS = 8
# Use PyMuPDF (fitz) for PDF by default: pypdf can explode memory (e.g. 150KB file -> 4GB)
# on PDFs with inline images or large content streams; PyMuPDF is much more memory-safe.
USE_PYPDF_FOR_PDF = False
# On Linux (Docker), run each PDF page extraction in a subprocess with a memory limit so one
# bad page cannot OOM the main API process (exit 137).
USE_SUBPROCESS_FOR_PDF_PAGE = sys.platform == "linux"
SUBPROCESS_PAGE_MEMORY_MB = 768  # per-page extraction subprocess limit
SUBPROCESS_COUNT_MEMORY_MB = 256  # page-count subprocess limit


def _get_pdf_page_count_worker(path: str, result_queue: multiprocessing.Queue, use_pymupdf: bool) -> None:
    """Run in subprocess: open PDF with memory limit, put page count in queue. Used only on Linux."""
    try:
        if sys.platform == "linux":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (SUBPROCESS_COUNT_MEMORY_MB * 1024 * 1024,) * 2)
        if use_pymupdf:
            import fitz as _fitz
            doc = _fitz.open(path)
            n = len(doc)
            doc.close()
        else:
            from pypdf import PdfReader as _PdfReader
            r = _PdfReader(path, strict=False)
            n = len(r.pages)
        result_queue.put(n)
    except Exception:
        result_queue.put(0)


def _extract_pdf_page_worker(
    path: str, page_index_0based: int, result_queue: multiprocessing.Queue, use_pymupdf: bool
) -> None:
    """Run in subprocess: open PDF with memory limit, extract one page text, put in queue. Used only on Linux."""
    try:
        if sys.platform == "linux":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (SUBPROCESS_PAGE_MEMORY_MB * 1024 * 1024,) * 2)
        if use_pymupdf:
            import fitz as _fitz
            doc = _fitz.open(path)
            page = doc[page_index_0based]
            text = page.get_text() or ""
            doc.close()
        else:
            from pypdf import PdfReader as _PdfReader
            r = _PdfReader(path, strict=False)
            page = r.pages[page_index_0based]
            text = page.extract_text() or ""
        result_queue.put(text)
    except Exception:
        result_queue.put("")


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

    def _iter_pdf_pages(self, file_path: Path):
        """
        Yield (page_number, page_text) one page at a time to bound memory.
        On Linux (Docker), uses a subprocess per page with memory limit so one bad page cannot OOM the API.
        Otherwise uses pypdf when USE_PYPDF_FOR_PDF else PyMuPDF in-process.
        """
        use_pymupdf = HAS_PYMUPDF and not USE_PYPDF_FOR_PDF
        if USE_SUBPROCESS_FOR_PDF_PAGE and (HAS_PYMUPDF or HAS_PYPDF):
            # Subprocess path: one bad page OOMs the child, not the API (Linux/Docker only)
            path_str = str(file_path)
            count_queue = multiprocessing.Queue()
            count_proc = multiprocessing.Process(
                target=_get_pdf_page_count_worker,
                args=(path_str, count_queue, use_pymupdf),
            )
            count_proc.start()
            count_proc.join(timeout=30)
            if count_proc.exitcode != 0:
                logger.warning("Subprocess page count failed, falling back to in-process PDF iteration")
            else:
                try:
                    page_count = count_queue.get_nowait()
                except Exception:
                    page_count = 0
                if page_count > 0:
                    for i in range(page_count):
                        crash_logger.write_progress("before_page_extract", {"page": i + 1})
                        sys.stdout.flush()
                        q = multiprocessing.Queue()
                        p = multiprocessing.Process(
                            target=_extract_pdf_page_worker,
                            args=(path_str, i, q, use_pymupdf),
                        )
                        p.start()
                        p.join(timeout=60)
                        if p.exitcode == 0:
                            try:
                                page_text = q.get_nowait() or ""
                            except Exception:
                                page_text = ""
                        else:
                            page_text = ""
                            if p.exitcode in (137, -9):
                                logger.warning(f"Page {i + 1} extraction OOM-killed (exit {p.exitcode}), skipping page")
                        crash_logger.write_progress("after_page_extract", {"page": i + 1})
                        sys.stdout.flush()
                        yield i + 1, page_text
                    return
        if USE_PYPDF_FOR_PDF or not HAS_PYMUPDF:
            if not HAS_PYPDF:
                raise RuntimeError("pypdf is required for PDF ingestion but is not installed")
            logger.debug("Using pypdf for page-by-page PDF extraction (memory-bounded)")
            # Write progress and flush so if OOM (exit 137) we see where we died in logs/crashes
            crash_logger.write_progress("before_pdf_open", {"path": str(file_path)})
            sys.stdout.flush()
            sys.stderr.flush()
            gc.collect()
            reader = PdfReader(str(file_path), strict=False)
            crash_logger.write_progress("after_pdf_open", {"path": str(file_path)})
            sys.stdout.flush()
            for index, page in enumerate(reader.pages, start=1):
                try:
                    crash_logger.write_progress("before_page_extract", {"page": index})
                    sys.stdout.flush()
                    page_text = page.extract_text() or ""
                    crash_logger.write_progress("after_page_extract", {"page": index})
                    sys.stdout.flush()
                    yield index, page_text
                    del page
                    if index % 5 == 0:
                        gc.collect()
                except Exception as e:
                    logger.warning(f"Error extracting page {index}: {e}")
                    yield index, ""
        else:
            logger.debug("Using PyMuPDF for page-by-page PDF extraction (memory-bounded)")
            doc = None
            try:
                doc = fitz.open(str(file_path))
                for index in range(len(doc)):
                    try:
                        crash_logger.write_progress("before_page_extract", {"page": index + 1})
                        sys.stdout.flush()
                        page = doc[index]
                        page_text = page.get_text() or ""
                        crash_logger.write_progress("after_page_extract", {"page": index + 1})
                        sys.stdout.flush()
                        yield index + 1, page_text
                        del page
                        if (index + 1) % 5 == 0:
                            gc.collect()
                    except Exception as e:
                        logger.warning(f"Error extracting page {index + 1}: {e}")
                        yield index + 1, ""
            finally:
                if doc:
                    try:
                        doc.close()
                        del doc
                    except Exception:
                        pass
                gc.collect()

    def _extract_all_text(self, file_path: Path, file_type: str) -> tuple[str, int]:
        """
        Extract all text from document synchronously.
        Returns: (full_text, page_count). Used for non-PDF or when not using page-by-page.
        """
        if file_type == "pdf":
            if HAS_PYMUPDF and not USE_PYPDF_FOR_PDF:
                logger.debug("Extracting text using PyMuPDF")
                doc = None
                try:
                    doc = fitz.open(str(file_path))
                    total_pages = len(doc)
                    logger.info(f"Extracting text from {total_pages} pages")
                    full_text = ""
                    for index in range(total_pages):
                        try:
                            page = doc[index]
                            page_text = page.get_text() or ""
                            full_text += page_text + "\n"
                            del page
                            if (index + 1) % 10 == 0:
                                gc.collect()
                        except Exception as e:
                            logger.warning(f"Error extracting page {index + 1}: {e}")
                    return full_text, total_pages
                finally:
                    if doc:
                        try:
                            doc.close()
                            del doc
                        except Exception:
                            pass
                    gc.collect()
            else:
                logger.debug("Extracting text using pypdf")
                reader = PdfReader(str(file_path), strict=False)
                total_pages = len(reader.pages)
                full_text = ""
                for index, page in enumerate(reader.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                        full_text += page_text + "\n"
                        del page
                        if index % 10 == 0:
                            gc.collect()
                    except Exception as e:
                        logger.warning(f"Error extracting page {index}: {e}")
                return full_text, total_pages
        else:
            full_text = file_path.read_text(encoding="utf-8")
            return full_text, 1

    def _chunk_text(self, text: str, chunk_words: int, overlap_words: int) -> list[str]:
        """
        Create all chunks from text synchronously.
        Simple sliding window approach - instant operation.
        """
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
        
        return chunks

    async def _update_document_status(self, session: AsyncSession, document: Document, status: str, message: str) -> None:
        """Helper to update document status with error handling."""
        try:
            await session.refresh(document)
            document.status = status
            document.meta_json = message[:1000]  # Limit message length
            await session.commit()
            logger.info(f"Document {document.id} status updated to {status}: {message[:100]}")
        except Exception as e:
            logger.error(f"Failed to update document status: {e}", exc_info=True)

    async def _ingest_pdf_page_by_page(
        self,
        session: AsyncSession,
        document: Document,
        config: WorkspaceConfig,
        file_path: Path,
        chunk_words: int,
        overlap_words: int,
        step_info: dict,
        total_start: float,
    ) -> None:
        """
        Ingest PDF page-by-page: for each page extract text -> chunk -> embed (small batch) -> insert -> commit.
        Keeps memory bounded (never holds full doc + all chunks + all embeddings).
        """
        chunk_index = 0
        page_count = 0
        total_chunks = 0
        
        await self._update_document_status(
            session, document, "processing",
            "Processing PDF page-by-page (memory-bounded)..."
        )
        logger.info("[PDF] Memory-bounded page-by-page ingestion started")
        crash_logger.write_progress("ingest_pdf_start", {"document_id": str(document.id)})
        sys.stdout.flush()
        sys.stderr.flush()
        
        try:
            for page_number, page_text in self._iter_pdf_pages(file_path):
                page_count = page_number
                if not page_text or not page_text.strip():
                    continue
                
                # Chunk this page's text only (no accumulation)
                page_chunks = self._chunk_text(page_text, chunk_words, overlap_words)
                del page_text
                gc.collect()
                
                if not page_chunks:
                    continue
                
                # Process in small batches to bound memory
                for i in range(0, len(page_chunks), INGEST_BATCH_CHUNKS):
                    batch = page_chunks[i:i + INGEST_BATCH_CHUNKS]
                    
                    if not self._check_memory_safe():
                        used_gb, _ = self._check_memory()
                        raise MemoryError(
                            f"Memory threshold ({MEMORY_THRESHOLD_GB}GB) exceeded at page {page_count}: {used_gb:.2f}GB"
                        )
                    
                    crash_logger.write_progress(
                        "before_embed_batch",
                        {"page": page_count, "batch_size": len(batch)}
                    )
                    sys.stdout.flush()
                    # Embed this batch only (small batch = low memory)
                    batch_embeddings = await self.embedding_service.embed_texts(
                        batch, config.embedding_model, batch_size=len(batch)
                    )
                    crash_logger.write_progress("after_embed_batch", {"page": page_count})
                    sys.stdout.flush()
                    if len(batch_embeddings) != len(batch):
                        raise RuntimeError(f"Embedding count mismatch: {len(batch_embeddings)} vs {len(batch)}")
                    
                    # Build chunk objects and insert
                    chunk_objects = [
                        DocumentChunk(
                            business_id=document.business_id,
                            workspace_id=document.workspace_id,
                            document_id=document.id,
                            chunk_index=chunk_index + j,
                            page_number=page_number,
                            content=chunk,
                            embedding=emb,
                        )
                        for j, (chunk, emb) in enumerate(zip(batch, batch_embeddings))
                    ]
                    session.add_all(chunk_objects)
                    await session.commit()
                    new_total = total_chunks + len(chunk_objects)
                    crash_logger.write_progress("after_commit", {"page": page_count, "total_chunks": new_total})
                    sys.stdout.flush()
                    
                    chunk_index += len(chunk_objects)
                    total_chunks = new_total
                    
                    del batch, batch_embeddings, chunk_objects
                    gc.collect()
                    
                    # Progress log after each batch so we have logs if killed
                    crash_logger.write_progress(
                        "after_batch",
                        {"page": page_count, "total_chunks": total_chunks}
                    )
                    sys.stdout.flush()
                
                del page_chunks
                gc.collect()
                
                if page_count % 5 == 0:
                    used_gb, _ = self._check_memory()
                    await self._update_document_status(
                        session, document, "processing",
                        f"Page {page_count}: {total_chunks} chunks stored. Memory: {used_gb:.2f}GB"
                    )
                    logger.info(f"[PDF] Page {page_count}: {total_chunks} chunks. Memory: {used_gb:.2f}GB")
            
            total_time = time.time() - total_start
            if total_chunks == 0:
                await self._update_document_status(session, document, "empty", "No chunks could be created")
                await session.refresh(document)
                document.status = "empty"
                document.indexed_at = datetime.utcnow()
                await session.commit()
                logger.warning(f"[PDF] Document {document.id} produced no chunks")
                return
            
            await session.refresh(document)
            document.status = "indexed"
            document.indexed_at = datetime.utcnow()
            document.meta_json = (
                f"Indexed {total_chunks} chunks from {page_count} pages (page-by-page). Time: {total_time:.2f}s"
            )
            await session.commit()
            logger.info(
                f"[PDF] Document {document.id} indexed: {page_count} pages, {total_chunks} chunks in {total_time:.2f}s"
            )
            
        except Exception as e:
            step_info["page_count"] = page_count
            step_info["total_chunks"] = total_chunks
            logger.error(f"[PDF] Page-by-page ingestion failed: {e}", exc_info=True)
            crash_logger.log_crash(
                e, type(e), e.__traceback__,
                context={
                    "operation": "document_ingestion_pdf_page_by_page",
                    "document_id": str(document.id),
                    "document_filename": document.filename,
                },
                additional_info=step_info
            )
            await self._update_document_status(
                session, document, "failed",
                f"{type(e).__name__}: {str(e)[:300]}. Check logs/crashes/."
            )
            raise

    async def ingest_document(
        self,
        session: AsyncSession,
        document: Document,
        config: WorkspaceConfig,
        chunk_words_override: int | None = None,
        overlap_words_override: int | None = None,
    ) -> None:
        """
        Ingest document using ChromaDB-style batch operations:
        1. Extract all text (sync)
        2. Create all chunks (sync)
        3. Generate ALL embeddings in batch
        4. Insert ALL chunks in single transaction
        """
        total_start = time.time()
        step_info = {}  # Track progress for crash logs
        crash_logger.write_progress(
            "ingest_document_start",
            {"document_id": str(document.id), "filename": document.filename or "", "file_type": document.file_type or ""}
        )
        sys.stdout.flush()
        sys.stderr.flush()

        logger.info(
            f"[INGEST START] document_id={document.id}, "
            f"filename={document.filename}, file_type={document.file_type}"
        )

        try:
            # INITIALIZATION STEP
            logger.info("[STEP 0] Initialization and validation...")
            await self._update_document_status(session, document, "processing", "Initializing document processing...")
            
            file_path = Path(document.storage_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {document.storage_path}")
            
            step_info["file_exists"] = True
            step_info["file_path"] = str(file_path)

            # Validate file size upfront
            self._validate_file_size(file_path)
            step_info["file_size_validated"] = True

            # Check initial memory
            used_gb, percent = self._check_memory()
            step_info["initial_memory_gb"] = used_gb
            step_info["initial_memory_percent"] = percent
            
            if not self._check_memory_safe():
                raise MemoryError(f"Memory usage too high at start: {used_gb:.2f}GB ({percent:.1f}%)")

            chunk_words = chunk_words_override or config.chunk_words
            overlap_words = overlap_words_override or config.overlap_words
            step_info["chunk_words"] = chunk_words
            step_info["overlap_words"] = overlap_words
            
            logger.info(f"[STEP 0] Initialization complete. Memory: {used_gb:.2f}GB")

            # PDF: use memory-bounded page-by-page ingestion to avoid OOM
            if document.file_type == "pdf":
                await self._ingest_pdf_page_by_page(
                    session, document, config, file_path,
                    chunk_words, overlap_words, step_info, total_start
                )
                return

            # Non-PDF or fallback: full extract -> chunk -> embed -> insert
            # STEP 1: Extract all text synchronously (like ChromaDB)
            logger.info("[STEP 1] Extracting all text from document...")
            await self._update_document_status(session, document, "processing", "Step 1/4: Extracting text from document...")
            
            extraction_start = time.time()
            try:
                full_text, page_count = self._extract_all_text(file_path, document.file_type)
                extraction_time = time.time() - extraction_start
                step_info["extraction_completed"] = True
                step_info["extraction_time"] = extraction_time
                step_info["text_length"] = len(full_text)
                step_info["page_count"] = page_count
                
                used_gb, percent = self._check_memory()
                step_info["memory_after_extraction_gb"] = used_gb
                
                logger.info(f"[STEP 1] ✅ Text extraction completed: {len(full_text)} chars, {page_count} pages in {extraction_time:.2f}s. Memory: {used_gb:.2f}GB")
                
                if not self._check_memory_safe():
                    raise MemoryError(f"Memory exceeded after extraction: {used_gb:.2f}GB")
                    
            except Exception as e:
                step_info["extraction_failed"] = True
                step_info["extraction_error"] = str(e)
                logger.error(f"[STEP 1] ❌ Text extraction failed: {e}", exc_info=True)
                crash_logger.log_crash(
                    e, type(e), e.__traceback__,
                    context={
                        "operation": "document_ingestion_step1_extraction",
                        "document_id": str(document.id),
                        "document_filename": document.filename,
                    },
                    additional_info=step_info
                )
                raise
            
            # Clear text from memory if very large (but keep for chunking)
            if len(full_text) > 10_000_000:  # > 10MB text
                logger.warning("Large text extracted, forcing GC before chunking")
                gc.collect()

            # STEP 2: Create all chunks synchronously (instant, like ChromaDB)
            logger.info("[STEP 2] Creating chunks from text...")
            await self._update_document_status(session, document, "processing", f"Step 2/4: Creating chunks from {page_count} pages...")
            
            chunking_start = time.time()
            try:
                all_chunks = self._chunk_text(full_text, chunk_words, overlap_words)
                chunking_time = time.time() - chunking_start
                step_info["chunking_completed"] = True
                step_info["chunking_time"] = chunking_time
                step_info["chunk_count"] = len(all_chunks)
                
                used_gb, percent = self._check_memory()
                step_info["memory_after_chunking_gb"] = used_gb
                
                logger.info(f"[STEP 2] ✅ Chunking completed: {len(all_chunks)} chunks in {chunking_time:.3f}s. Memory: {used_gb:.2f}GB")
                
                # Clear full_text from memory now that we have chunks
                del full_text
                gc.collect()
                
            except Exception as e:
                step_info["chunking_failed"] = True
                step_info["chunking_error"] = str(e)
                logger.error(f"[STEP 2] ❌ Chunking failed: {e}", exc_info=True)
                crash_logger.log_crash(
                    e, type(e), e.__traceback__,
                    context={
                        "operation": "document_ingestion_step2_chunking",
                        "document_id": str(document.id),
                        "text_length": len(full_text) if 'full_text' in locals() else 0,
                    },
                    additional_info=step_info
                )
                raise
            
            if not all_chunks:
                logger.warning(f"[STEP 2] Document {document.id} produced no chunks - marking as empty")
                await self._update_document_status(session, document, "empty", "No chunks could be created from document")
                await session.refresh(document)
                document.indexed_at = datetime.utcnow()
                await session.commit()
                return

            # Check memory before embedding generation
            used_gb, percent = self._check_memory()
            if not self._check_memory_safe():
                raise MemoryError(
                    f"Memory threshold ({MEMORY_THRESHOLD_GB}GB) exceeded before embedding: {used_gb:.2f}GB. "
                    f"Stopping to prevent crash."
                )

            # STEP 3: Generate ALL embeddings in batch (like ChromaDB)
            logger.info(f"[STEP 3] Generating embeddings for {len(all_chunks)} chunks in batch...")
            await self._update_document_status(session, document, "processing", f"Step 3/4: Generating embeddings for {len(all_chunks)} chunks...")
            
            embedding_start = time.time()
            try:
                # Generate all embeddings at once (or in large batches if needed)
                all_embeddings = await self.embedding_service.embed_texts(
                    all_chunks, 
                    config.embedding_model, 
                    batch_size=None  # Let embedding service decide optimal batch size
                )
                
                embedding_time = time.time() - embedding_start
                step_info["embedding_completed"] = True
                step_info["embedding_time"] = embedding_time
                step_info["embedding_count"] = len(all_embeddings)
                
                used_gb, percent = self._check_memory()
                step_info["memory_after_embedding_gb"] = used_gb
                
                logger.info(f"[STEP 3] ✅ Embedding generation completed: {len(all_embeddings)} embeddings in {embedding_time:.2f}s. Memory: {used_gb:.2f}GB")

                if len(all_embeddings) != len(all_chunks):
                    raise RuntimeError(
                        f"Embedding count mismatch: {len(all_embeddings)} embeddings for {len(all_chunks)} chunks"
                    )
                    
            except Exception as e:
                step_info["embedding_failed"] = True
                step_info["embedding_error"] = str(e)
                logger.error(f"[STEP 3] ❌ Embedding generation failed: {e}", exc_info=True)
                crash_logger.log_crash(
                    e, type(e), e.__traceback__,
                    context={
                        "operation": "document_ingestion_step3_embedding",
                        "document_id": str(document.id),
                        "chunk_count": len(all_chunks) if 'all_chunks' in locals() else 0,
                    },
                    additional_info=step_info
                )
                raise

            # Check memory before database insert
            used_gb, percent = self._check_memory()
            if not self._check_memory_safe():
                raise MemoryError(
                    f"Memory threshold ({MEMORY_THRESHOLD_GB}GB) exceeded before insert: {used_gb:.2f}GB. "
                    f"Stopping to prevent crash."
                )

            # STEP 4: Insert ALL chunks in single batch transaction (like ChromaDB)
            logger.info(f"[STEP 4] Inserting {len(all_chunks)} chunks in single batch transaction...")
            await self._update_document_status(session, document, "processing", f"Step 4/4: Storing {len(all_chunks)} chunks in database...")
            
            insert_start = time.time()
            try:
                # Prepare all chunk objects
                logger.info(f"[STEP 4] Preparing {len(all_chunks)} chunk objects...")
                chunk_objects = [
                    DocumentChunk(
                        business_id=document.business_id,
                        workspace_id=document.workspace_id,
                        document_id=document.id,
                        chunk_index=index,
                        page_number=None,  # We don't track page numbers in batch mode
                        content=chunk,
                        embedding=embedding,
                    )
                    for index, (chunk, embedding) in enumerate(zip(all_chunks, all_embeddings))
                ]
                logger.info(f"[STEP 4] Chunk objects prepared, adding to session...")
                
                # Single batch insert - like ChromaDB's collection.add()
                session.add_all(chunk_objects)
                logger.info(f"[STEP 4] Chunks added to session, committing transaction...")
                await session.commit()
                
                insert_time = time.time() - insert_start
                step_info["insert_completed"] = True
                step_info["insert_time"] = insert_time
                step_info["chunks_inserted"] = len(chunk_objects)
                
                used_gb, percent = self._check_memory()
                step_info["memory_after_insert_gb"] = used_gb
                
                logger.info(f"[STEP 4] ✅ Database insert completed: {len(chunk_objects)} chunks in {insert_time:.2f}s. Memory: {used_gb:.2f}GB")
                
            except Exception as e:
                step_info["insert_failed"] = True
                step_info["insert_error"] = str(e)
                logger.error(f"[STEP 4] ❌ Database insert failed: {e}", exc_info=True)
                try:
                    await session.rollback()
                    logger.info("[STEP 4] Transaction rolled back")
                except Exception as rollback_err:
                    logger.error(f"[STEP 4] Rollback failed: {rollback_err}")
                
                crash_logger.log_crash(
                    e, type(e), e.__traceback__,
                    context={
                        "operation": "document_ingestion_step4_database_insert",
                        "document_id": str(document.id),
                        "chunk_count": len(all_chunks) if 'all_chunks' in locals() else 0,
                        "embedding_count": len(all_embeddings) if 'all_embeddings' in locals() else 0,
                    },
                    additional_info=step_info
                )
                raise

            # Store chunk count before clearing
            chunk_count = len(chunk_objects)
            
            # Clear from memory
            del all_chunks, all_embeddings, chunk_objects
            gc.collect()

            # Final status update
            total_time = time.time() - total_start
            logger.info(
                f"Document processed successfully: {page_count} pages, {chunk_count} chunks "
                f"in {total_time:.2f}s (Extract: {extraction_time:.2f}s, Chunk: {chunking_time:.3f}s, "
                f"Embed: {embedding_time:.2f}s, Insert: {insert_time:.2f}s)"
            )

            await session.refresh(document)
            document.status = "indexed"
            document.indexed_at = datetime.utcnow()
            document.meta_json = (
                f"Successfully indexed: {chunk_count} chunks from {page_count} pages. "
                f"Total time: {total_time:.2f}s"
            )
            await session.commit()

            logger.info(
                f"Document {document.id} successfully indexed. "
                f"Total time: {total_time:.2f}s"
            )
                
        except MemoryError as mem_error:
            # Log crash with detailed context
            used_gb, percent = self._check_memory()
            step_info["final_memory_gb"] = used_gb
            step_info["final_memory_percent"] = percent
            
            crash_log_file = crash_logger.log_crash(
                mem_error, type(mem_error), mem_error.__traceback__,
                context={
                    "operation": "document_ingestion_memory_error",
                    "document_id": str(document.id),
                    "document_filename": document.filename,
                    "file_type": document.file_type,
                },
                additional_info=step_info
            )
            
            logger.error(f"[CRASH] Out of memory while processing document: {mem_error}")
            logger.error(f"[CRASH] Crash log saved to: {crash_log_file}")
            logger.error(f"[CRASH] Final memory: {used_gb:.2f}GB ({percent:.1f}%)")
            
            await self._update_document_status(
                session, document, "failed",
                f"Memory error: {str(mem_error)[:200]}. Check logs/crashes/ for details."
            )
            raise
        except Exception as e:
            # Log crash with full context
            is_critical = isinstance(e, (MemoryError, SystemError, KeyboardInterrupt))
            
            if is_critical:
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
                )
            else:
                crash_logger.log_error(
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
                        f"Check logs/crashes/ for detailed {'crash' if is_critical else 'error'} report."
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
