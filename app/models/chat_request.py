import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ChatRequest(Base):
    __tablename__ = "chat_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("businesses.id"))
    workspace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workspaces.id"))
    user_id: Mapped[str] = mapped_column(String(255))
    query_text: Mapped[str] = mapped_column(String)
    retrieved_chunk_ids: Mapped[str] = mapped_column(String)

    response = relationship("ChatResponse", back_populates="request", uselist=False)
