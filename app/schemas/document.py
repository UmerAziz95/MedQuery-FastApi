import uuid

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    status: str

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    business_client_id: str
    workspace_id: str


class DocumentChunkOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    page_number: int | None = None
    content: str

    class Config:
        from_attributes = True
