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
        try:
            file_path = Path(document.storage_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {document.storage_path}")
            
            # Phase 1: Extract text from PDF
            logger.debug(f"Loading text from file: {file_path}")
            extraction_start = time.time()
            text_pages = self._load_text(file_path, document.file_type)
            chunk_words = chunk_words_override or config.chunk_words
            overlap_words = overlap_words_override or config.overlap_words
            chunks = []
            
            page_count = 0
            for page_number, page_text in text_pages:
                page_count += 1
                if page_count % 10 == 0:  # Log every 10 pages for large PDFs
                    logger.debug(f"Processing page {page_number}...")
                for chunk in self._chunk_text(page_text, chunk_words, overlap_words):
                    chunks.append((page_number, chunk))
            
            extraction_time = time.time() - extraction_start
            per_page_time = extraction_time / page_count if page_count > 0 else 0
            logger.info(
                f"Document processed: {page_count} pages, {len(chunks)} chunks created "
                f"in {extraction_time:.2f}s ({per_page_time:.3f}s per page)"
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
            
            embedding_start = time.time()
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

            # Phase 3: Store in database
            logger.debug(f"Storing {len(chunks)} document chunks in database")
            db_start = time.time()
            session.add_all(
                [
                    DocumentChunk(
                        business_id=document.business_id,
                        workspace_id=document.workspace_id,
                        document_id=document.id,
                        chunk_index=index,
                        page_number=page,
                        content=content,
                        embedding=embedding,
                    )
                    for index, ((page, content), embedding) in enumerate(zip(chunks, embeddings))
                ]
            )
            document.status = "indexed"
            document.indexed_at = datetime.utcnow()
            await session.commit()
            db_time = time.time() - db_start
            total_time = time.time() - total_start
            
            logger.info(
                f"Document {document.id} successfully indexed with {len(chunks)} chunks. "
                f"Database write: {db_time:.2f}s. Total time: {total_time:.2f}s "
                f"(Extraction: {extraction_time:.2f}s, Embedding: {embedding_time:.2f}s, DB: {db_time:.2f}s)"
            )
        except Exception as e:
            logger.error(
                f"Error ingesting document {document.id} ({document.filename}): {type(e).__name__}: {str(e)}",
                exc_info=True
            )
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
