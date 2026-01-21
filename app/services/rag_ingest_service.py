import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

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
                    # Extract pages one at a time
                    def extract_pages_incremental():
                        """Extract pages one at a time to save memory."""
                        reader = PdfReader(str(file_path))
                        for index, page in enumerate(reader.pages, start=1):
                            try:
                                text = page.extract_text() or ""
                                yield index, text
                            except Exception as e:
                                logger.warning(f"Error extracting page {index}: {e}")
                                yield index, ""
                    
                    # Process each page as we extract it
                    for page_number, page_text in extract_pages_incremental():
                        page_count += 1
                        if page_count % 10 == 0:
                            logger.debug(f"Processing page {page_number}...")
                        
                        # Chunk the page text
                        if len(page_text) > 10000:
                            # Large pages: chunk in thread pool
                            chunk_list = await loop.run_in_executor(
                                None,
                                lambda pt=page_text: self._chunk_text(pt, chunk_words, overlap_words)
                            )
                        else:
                            chunk_list = self._chunk_text(page_text, chunk_words, overlap_words)
                        
                        # Add chunks for this page
                        for chunk in chunk_list:
                            chunks.append((page_number, chunk))
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
            
            # Phase 2: Generate embeddings with error handling
            embedding_start = time.time()
            try:
                embeddings = await self.embedding_service.embed_texts(
                    [chunk for _, chunk in chunks], config.embedding_model
                )
                embedding_time = time.time() - embedding_start
                
                if is_small_file:
                    logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
                else:
                    logger.info(
                        f"Generated {len(embeddings)} embeddings successfully in {embedding_time:.2f}s "
                        f"({embedding_time/len(embeddings):.3f}s per embedding)"
                    )
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

            # Phase 3: Store in database with optimized batch inserts
            logger.debug(f"Storing {len(chunks)} document chunks in database")
            db_start = time.time()
            
            try:
                # Optimize batch size based on chunk count
                # Small files: single insert, Medium: 100 chunks/batch, Large: 200 chunks/batch
                if len(chunks) <= 50:
                    # Small files: single insert (fastest)
                    batch_size = len(chunks)
                elif len(chunks) <= 500:
                    # Medium files: 100 chunks per batch
                    batch_size = 100
                else:
                    # Large files: 200 chunks per batch (fewer commits = faster)
                    batch_size = 200
                
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                saved_chunks = 0
                
                for batch_idx in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[batch_idx:batch_idx + batch_size]
                    batch_embeddings = embeddings[batch_idx:batch_idx + batch_size]
                    batch_num = (batch_idx // batch_size) + 1
                    
                    try:
                        # Create all chunk objects
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
                        
                        # Bulk insert for better performance
                        session.add_all(chunk_objects)
                        
                        # For small files, commit once. For larger files, commit every batch for progress visibility
                        if len(chunks) <= 50:
                            # Small files: commit once at the end
                            pass
                        else:
                            # Larger files: commit each batch for progress tracking
                            await session.flush()  # Flush instead of commit for better performance
                            if batch_num % 5 == 0 or batch_num == total_batches:
                                # Commit every 5 batches or on last batch
                                await session.commit()
                            saved_chunks += len(batch_chunks)
                            if batch_num % 5 == 0:
                                logger.debug(f"Stored batch {batch_num}/{total_batches} ({saved_chunks}/{len(chunks)} chunks saved)")
                    except Exception as batch_error:
                        logger.error(f"Error saving batch {batch_num}: {batch_error}", exc_info=True)
                        await session.rollback()
                        # Try to save what we can - continue with next batch
                        if saved_chunks == 0:
                            # If first batch fails, mark as failed
                            raise
                        logger.warning(f"Continuing after batch {batch_num} error, {saved_chunks} chunks already saved")
                
                # Final commit for all chunks
                await session.commit()
                saved_chunks = len(chunks)
                
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
