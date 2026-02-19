import uuid

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    status: str
    chunk_count: int | None = None
    indexed_at: str | None = None
    meta_json: str | None = None

    class Config:
        from_attributes = True


class DocumentStatusResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    status: str
    chunk_count: int
    indexed_at: str | None = None
    meta_json: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    business_client_id: str
    workspace_id: str
    chunk_count: int | None = None


class DocumentChunkOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    page_number: int | None = None
    content: str

    class Config:
        from_attributes = True
