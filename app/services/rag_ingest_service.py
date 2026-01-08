import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import Document, DocumentChunk, WorkspaceConfig
from app.services.embedding_service import EmbeddingService


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
        file_path = Path(document.storage_path)
        text_pages = self._load_text(file_path, document.file_type)
        chunk_words = chunk_words_override or config.chunk_words
        overlap_words = overlap_words_override or config.overlap_words
        chunks = []
        for page_number, page_text in text_pages:
            for chunk in self._chunk_text(page_text, chunk_words, overlap_words):
                chunks.append((page_number, chunk))

        embeddings = await self.embedding_service.embed_texts(
            [chunk for _, chunk in chunks], config.embedding_model
        )

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
