import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ChatResponse(Base):
    __tablename__ = "chat_responses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_requests.id"))
    answer_text: Mapped[str] = mapped_column(String)
    sources_json: Mapped[str] = mapped_column(String)
    model_used: Mapped[str] = mapped_column(String(255))
    tokens_json: Mapped[str] = mapped_column(String)

    request = relationship("ChatRequest", back_populates="response")
