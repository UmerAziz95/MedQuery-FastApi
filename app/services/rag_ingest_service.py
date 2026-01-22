import asyncio
import gc
import logging
import os
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
    import fitz  # PyMuPDF - more memory efficient
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    from pypdf import PdfReader
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import Document, DocumentChunk, WorkspaceConfig
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


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
        """Check if memory usage is safe (< 3.5GB used)."""
        if not HAS_PSUTIL:
            return True  # Can't check, assume safe
        used_gb, percent = self._check_memory()
        if used_gb > 3.5:  # More than 3.5GB used
            logger.warning(f"High memory usage detected: {used_gb:.2f}GB ({percent:.1f}%)")
            gc.collect()  # Force cleanup
            return False
        return True

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
        
        chunks = []
        embeddings = []
        
        try:
            file_path = Path(document.storage_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {document.storage_path}")
            
            # Phase 1: Extract text from PDF (process incrementally to avoid memory issues)
            logger.debug(f"Loading text from file: {file_path}")
            extraction_start = time.time()
            
            chunk_words = chunk_words_override or config.chunk_words
            overlap_words = overlap_words_override or config.overlap_words
            
            # Process PDF pages incrementally to avoid loading all into memory
            # This prevents OOM (Out of Memory) crashes
            loop = asyncio.get_event_loop()
            chunking_start = time.time()
            page_count = 0
            
            # Process pages one at a time to avoid memory spikes
            def process_single_page(page_data):
                """Process a single page: extract text and chunk it."""
                page_number, page_text = page_data
                return page_number, self._chunk_text(page_text, chunk_words, overlap_words)
            
            # Load and process pages incrementally
            try:
                # For PDFs, process page by page to avoid memory issues
                if document.file_type == "pdf":
                    # Extract and process PDF entirely in thread pool to avoid memory issues
                    def extract_and_process_pdf():
                        """Extract all pages and chunk them in thread pool to save memory."""
                        doc = None
                        try:
                            logger.debug(f"Opening PDF file: {file_path}")
                            # Force garbage collection before opening PDF
                            gc.collect()
                            
                            all_chunks = []
                            
                            # Use PyMuPDF (fitz) if available - much more memory efficient
                            if HAS_PYMUPDF:
                                logger.debug("Using PyMuPDF (fitz) for memory-efficient PDF processing")
                                doc = fitz.open(str(file_path))
                                total_pages = len(doc)
                                logger.debug(f"PDF has {total_pages} pages")
                                
                                # Process pages one at a time with aggressive cleanup
                                for index in range(total_pages):
                                    try:
                                        page = doc[index]
                                        text = page.get_text() or ""
                                        
                                        # Immediately delete page reference
                                        del page
                                        
                                        if not text.strip():
                                            logger.debug(f"Page {index + 1} has no text, skipping")
                                            del text
                                            gc.collect()
                                            continue
                                        
                                        # Chunk immediately to free text memory
                                        page_chunks = self._chunk_text(text, chunk_words, overlap_words)
                                        del text
                                        
                                        for chunk in page_chunks:
                                            all_chunks.append((index + 1, chunk))
                                        
                                        # Force garbage collection every page
                                        gc.collect()
                                        
                                        if (index + 1) % 5 == 0:
                                            logger.debug(f"Processed {index + 1}/{total_pages} pages, {len(all_chunks)} chunks so far")
                                    except Exception as e:
                                        logger.warning(f"Error extracting page {index + 1}: {e}")
                                        gc.collect()
                            else:
                                # Fallback to pypdf
                                logger.debug("Using pypdf for PDF processing")
                                reader = PdfReader(str(file_path), strict=False)
                                total_pages = len(reader.pages)
                                logger.debug(f"PDF has {total_pages} pages")
                                
                                for index in range(1, total_pages + 1):
                                    try:
                                        page = reader.pages[index - 1]
                                        text = page.extract_text() or ""
                                        del page
                                        
                                        if not text.strip():
                                            logger.debug(f"Page {index} has no text, skipping")
                                            del text
                                            gc.collect()
                                            continue
                                        
                                        page_chunks = self._chunk_text(text, chunk_words, overlap_words)
                                        del text
                                        
                                        for chunk in page_chunks:
                                            all_chunks.append((index, chunk))
                                        
                                        gc.collect()
                                        
                                        if index % 5 == 0:
                                            logger.debug(f"Processed {index}/{total_pages} pages, {len(all_chunks)} chunks so far")
                                    except Exception as e:
                                        logger.warning(f"Error extracting page {index}: {e}")
                                        gc.collect()
                            
                            logger.debug(f"PDF processing complete: {len(all_chunks)} chunks from {total_pages} pages")
                            return all_chunks, total_pages
                        except MemoryError as mem_err:
                            logger.error(f"Out of memory while reading PDF: {mem_err}")
                            gc.collect()
                            raise
                        except Exception as e:
                            logger.error(f"Error reading PDF: {type(e).__name__}: {e}", exc_info=True)
                            raise
                        finally:
                            # Explicitly clean up and force GC
                            if doc:
                                try:
                                    doc.close()
                                except:
                                    pass
                                del doc
                            gc.collect()
                    
                    # Run entire PDF processing in thread pool (isolates memory usage)
                    logger.debug("Processing PDF in thread pool to avoid memory issues")
                    try:
                        chunks_list, page_count = await loop.run_in_executor(None, extract_and_process_pdf)
                        chunks = chunks_list
                    except MemoryError as mem_err:
                        logger.error(f"Memory error during PDF processing: {mem_err}")
                        document.status = "failed"
                        document.meta_json = f"Out of memory: PDF too large. File size: {file_path.stat().st_size / 1024:.1f}KB"
                        await session.commit()
                        raise
                else:
                    # For TXT files, process normally (they're smaller)
                    text_pages = await loop.run_in_executor(
                        None,
                        lambda: list(self._load_text(file_path, document.file_type))
                    )
                    for page_number, page_text in text_pages:
                        page_count += 1
                        for chunk in self._chunk_text(page_text, chunk_words, overlap_words):
                            chunks.append((page_number, chunk))
            except MemoryError as mem_error:
                logger.error(f"Out of memory while processing PDF: {mem_error}")
                document.status = "failed"
                document.meta_json = f"Memory error: PDF too large or complex. Try splitting the document."
                await session.commit()
                raise
            except Exception as extract_error:
                logger.error(f"Error extracting PDF: {extract_error}", exc_info=True)
                document.status = "failed"
                document.meta_json = f"PDF extraction error: {type(extract_error).__name__}: {str(extract_error)}"
                await session.commit()
                raise
            
            extraction_time = time.time() - extraction_start
            chunking_time = time.time() - chunking_start
            per_page_time = extraction_time / page_count if page_count > 0 else 0
            logger.info(
                f"Document processed: {page_count} pages, {len(chunks)} chunks created "
                f"in {extraction_time:.2f}s (extraction: {extraction_time - chunking_time:.2f}s, "
                f"chunking: {chunking_time:.2f}s, {per_page_time:.3f}s per page)"
            )

            if not chunks:
                logger.warning(f"Document {document.id} produced no chunks - marking as empty")
                document.status = "empty"
                document.indexed_at = datetime.utcnow()
                await session.commit()
                return

            # Calculate total text length for logging (only for larger files)
            num_chunks = len(chunks)
            is_small_file = num_chunks <= 50
            
            if not is_small_file:
                total_text_length = sum(len(chunk) for _, chunk in chunks)
                logger.info(
                    f"Generating embeddings for {num_chunks} chunks "
                    f"(total text length: {total_text_length:,} chars) using model: {config.embedding_model}"
                )
            else:
                logger.debug(f"Generating embeddings for {num_chunks} chunks (small file)")
            
            # Phase 2: Generate embeddings in small batches to minimize memory usage
            embedding_start = time.time()
            try:
                # Check memory before embedding
                if not self._check_memory_safe():
                    logger.warning("High memory usage before embedding, forcing GC")
                    gc.collect()
                
                # Process embeddings in small batches to avoid memory accumulation
                chunk_texts = [chunk for _, chunk in chunks]
                num_chunks = len(chunk_texts)
                
                # Use very small batches (25 chunks) to minimize memory
                embed_batch_size = 25
                embeddings = []
                
                logger.debug(f"Generating embeddings in batches of {embed_batch_size} to minimize memory")
                
                for i in range(0, num_chunks, embed_batch_size):
                    batch_texts = chunk_texts[i:i + embed_batch_size]
                    
                    # Check memory before each batch
                    if not self._check_memory_safe():
                        logger.warning(f"High memory before embedding batch {i // embed_batch_size + 1}, forcing GC")
                        gc.collect()
                    
                    # Generate embeddings for this batch
                    batch_embeddings = await self.embedding_service.embed_texts(
                        batch_texts, config.embedding_model, batch_size=len(batch_texts)
                    )
                    embeddings.extend(batch_embeddings)
                    
                    # Clear batch from memory immediately
                    del batch_texts, batch_embeddings
                    gc.collect()
                    
                    if (i // embed_batch_size + 1) % 10 == 0:
                        logger.debug(f"Generated embeddings for {len(embeddings)}/{num_chunks} chunks")
                
                # Clear chunk texts from memory
                del chunk_texts
                gc.collect()
                
                embedding_time = time.time() - embedding_start
                
                if is_small_file:
                    logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
                else:
                    logger.info(
                        f"Generated {len(embeddings)} embeddings successfully in {embedding_time:.2f}s "
                        f"({embedding_time/len(embeddings):.3f}s per embedding)"
                    )
                
                # Check memory after embedding
                used_gb, percent = self._check_memory()
                logger.debug(f"Memory after embeddings: {used_gb:.2f}GB ({percent:.1f}%)")
                
            except Exception as embed_error:
                logger.error(f"Embedding generation failed: {embed_error}", exc_info=True)
                document.status = "failed"
                document.meta_json = f"Embedding error: {type(embed_error).__name__}: {str(embed_error)}"
                await session.commit()
                raise

            if len(embeddings) != len(chunks):
                error_msg = f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
                logger.error(error_msg)
                document.status = "failed"
                document.meta_json = error_msg
                await session.commit()
                raise RuntimeError(error_msg)

            # Phase 3: Store in database with optimized batch inserts and memory monitoring
            logger.debug(f"Storing {len(chunks)} document chunks in database")
            db_start = time.time()
            
            try:
                # Use smaller batches to reduce memory pressure
                # Process in batches of 50 to minimize memory usage
                batch_size = 50
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                saved_chunks = 0
                
                for batch_idx in range(0, len(chunks), batch_size):
                    # Check memory before each batch
                    if not self._check_memory_safe():
                        logger.warning(f"High memory usage before batch {batch_idx // batch_size + 1}, forcing GC")
                        gc.collect()
                    
                    batch_chunks = chunks[batch_idx:batch_idx + batch_size]
                    batch_embeddings = embeddings[batch_idx:batch_idx + batch_size]
                    batch_num = (batch_idx // batch_size) + 1
                    
                    try:
                        # Create chunk objects for this batch
                        chunk_objects = [
                            DocumentChunk(
                                business_id=document.business_id,
                                workspace_id=document.workspace_id,
                                document_id=document.id,
                                chunk_index=batch_idx + index,
                                page_number=page,
                                content=content,
                                embedding=embedding,
                            )
                            for index, ((page, content), embedding) in enumerate(zip(batch_chunks, batch_embeddings))
                        ]
                        
                        # Add and commit immediately to free memory
                        session.add_all(chunk_objects)
                        await session.commit()
                        saved_chunks += len(batch_chunks)
                        
                        # Clear batch data from memory
                        del chunk_objects, batch_chunks, batch_embeddings
                        gc.collect()
                        
                        if batch_num % 10 == 0 or batch_num == total_batches:
                            logger.debug(f"Stored batch {batch_num}/{total_batches} ({saved_chunks}/{len(chunks)} chunks saved)")
                    except Exception as batch_error:
                        logger.error(f"Error saving batch {batch_num}: {batch_error}", exc_info=True)
                        await session.rollback()
                        if saved_chunks == 0:
                            raise
                        logger.warning(f"Continuing after batch {batch_num} error, {saved_chunks} chunks already saved")
                
                # Clear chunks and embeddings from memory
                del chunks, embeddings
                gc.collect()
                
                saved_chunks = saved_chunks  # Final count
                
                # Update document status to indexed
                document.status = "indexed"
                document.indexed_at = datetime.utcnow()
                await session.commit()
                
                db_time = time.time() - db_start
                total_time = time.time() - total_start
                
                logger.info(
                    f"Document {document.id} successfully indexed with {saved_chunks} chunks. "
                    f"Database write: {db_time:.2f}s. Total time: {total_time:.2f}s "
                    f"(Extraction: {extraction_time:.2f}s, Embedding: {embedding_time:.2f}s, DB: {db_time:.2f}s)"
                )
            except Exception as db_error:
                logger.error(f"Database error during chunk storage: {db_error}", exc_info=True)
                await session.rollback()
                document.status = "failed"
                document.meta_json = f"Database error: {type(db_error).__name__}: {str(db_error)}"
                await session.commit()
                raise
                
        except Exception as e:
            logger.error(
                f"Error ingesting document {document.id} ({document.filename}): {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            # Ensure status is updated even on error
            try:
                if document.status == "processing":
                    document.status = "failed"
                    document.meta_json = f"{type(e).__name__}: {str(e)}"
                    await session.commit()
            except Exception as status_error:
                logger.error(f"Failed to update document status: {status_error}")
            raise

    async def reindex_document(self, session: AsyncSession, document: Document, config: WorkspaceConfig) -> None:
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        document.status = "processing"
        await session.commit()
        await self.ingest_document(session, document, config)

    def store_upload(self, filename: str, data: bytes) -> str:
        os.makedirs(self.settings.file_storage_path, exist_ok=True)
        path = Path(self.settings.file_storage_path) / filename
        path.write_bytes(data)
        return str(path)

    def _load_text(self, file_path: Path, file_type: str) -> Iterable[tuple[int | None, str]]:
        if file_type == "pdf":
            reader = PdfReader(str(file_path))
            for index, page in enumerate(reader.pages, start=1):
                yield index, page.extract_text() or ""
        else:
            yield None, file_path.read_text(encoding="utf-8")

    def _chunk_text(self, text: str, chunk_words: int, overlap_words: int) -> list[str]:
        words = text.split()
        if overlap_words >= chunk_words:
            overlap_words = 0
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_words)
            chunk = " ".join(words[start:end])
            if chunk:
                chunks.append(chunk)
            start = end - overlap_words
            if start < 0:
                start = 0
        return chunks
