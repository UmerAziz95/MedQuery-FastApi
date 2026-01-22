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

        try:
            file_path = Path(document.storage_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {document.storage_path}")

            logger.debug(f"Loading text from file: {file_path}")
            extraction_start = time.time()

            chunk_words = chunk_words_override or config.chunk_words
            overlap_words = overlap_words_override or config.overlap_words

            chunk_count = 0
            chunk_index = 0
            page_count = 0
            embedding_batches = 0
            embed_batch_size = min(self.embedding_service.default_batch_size, 200)
            pending_chunks: list[tuple[int | None, str]] = []

            async def process_batch(batch_chunks: list[tuple[int | None, str]]) -> None:
                nonlocal chunk_index, chunk_count, embedding_batches

                if not batch_chunks:
                    return

                if not self._check_memory_safe():
                    logger.warning("High memory before embedding batch, forcing GC")
                    gc.collect()

                batch_texts = [chunk for _, chunk in batch_chunks]
                embedding_batches += 1
                batch_embeddings = await self.embedding_service.embed_texts(
                    batch_texts, config.embedding_model, batch_size=len(batch_texts)
                )

                if len(batch_embeddings) != len(batch_chunks):
                    raise RuntimeError(
                        f"Embedding count mismatch: {len(batch_embeddings)} embeddings for {len(batch_chunks)} chunks"
                    )

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

                try:
                    session.add_all(chunk_objects)
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

                chunk_count += len(batch_chunks)
                chunk_index += len(batch_chunks)

                del batch_texts, batch_embeddings, chunk_objects
                gc.collect()

            try:
                for page_number, chunk in self._iter_chunks(file_path, document.file_type, chunk_words, overlap_words):
                    page_count = max(page_count, page_number or 1)
                    pending_chunks.append((page_number, chunk))

                    if len(pending_chunks) >= embed_batch_size:
                        await process_batch(pending_chunks)
                        pending_chunks = []

                    if chunk_index and chunk_index % (embed_batch_size * 5) == 0:
                        logger.debug(f"Processed {chunk_index} chunks so far")

                if pending_chunks:
                    await process_batch(pending_chunks)
                    pending_chunks = []
            except MemoryError as mem_error:
                logger.error(f"Out of memory while processing document: {mem_error}")
                document.status = "failed"
                document.meta_json = "Memory error: PDF too large or complex. Try splitting the document."
                await session.commit()
                raise
            except Exception as process_error:
                logger.error(f"Error processing document: {process_error}", exc_info=True)
                document.status = "failed"
                document.meta_json = f"Processing error: {type(process_error).__name__}: {str(process_error)}"
                await session.commit()
                raise

            extraction_time = time.time() - extraction_start
            per_page_time = extraction_time / page_count if page_count > 0 else 0
            logger.info(
                f"Document processed: {page_count} pages, {chunk_count} chunks created "
                f"in {extraction_time:.2f}s ({per_page_time:.3f}s per page, {embedding_batches} embedding batches)"
            )

            if chunk_count == 0:
                logger.warning(f"Document {document.id} produced no chunks - marking as empty")
                document.status = "empty"
                document.indexed_at = datetime.utcnow()
                await session.commit()
                return

            document.status = "indexed"
            document.indexed_at = datetime.utcnow()
            await session.commit()

            total_time = time.time() - total_start
            logger.info(
                f"Document {document.id} successfully indexed with {chunk_count} chunks. "
                f"Total time: {total_time:.2f}s (Extraction+Embedding+DB streaming)"
            )
                
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

    def _iter_chunks(
        self, file_path: Path, file_type: str, chunk_words: int, overlap_words: int
    ) -> Iterable[tuple[int | None, str]]:
        if file_type == "pdf":
            doc = None
            try:
                if HAS_PYMUPDF:
                    logger.debug("Using PyMuPDF (fitz) for PDF processing")
                    doc = fitz.open(str(file_path))
                    for index in range(len(doc)):
                        page_number = index + 1
                        try:
                            text = doc[index].get_text() or ""
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_number}: {e}")
                            continue
                        for chunk in self._chunk_text(text, chunk_words, overlap_words):
                            yield page_number, chunk
                else:
                    logger.debug("Using pypdf for PDF processing")
                    reader = PdfReader(str(file_path), strict=False)
                    for index, page in enumerate(reader.pages, start=1):
                        try:
                            text = page.extract_text() or ""
                        except Exception as e:
                            logger.warning(f"Error extracting page {index}: {e}")
                            continue
                        for chunk in self._chunk_text(text, chunk_words, overlap_words):
                            yield index, chunk
            finally:
                if doc:
                    try:
                        doc.close()
                    except Exception:
                        pass
                gc.collect()
        else:
            text = file_path.read_text(encoding="utf-8")
            for chunk in self._chunk_text(text, chunk_words, overlap_words):
                yield None, chunk

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
