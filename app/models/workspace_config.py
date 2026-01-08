import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class WorkspaceConfig(Base):
    __tablename__ = "workspace_config"
    __table_args__ = (
        UniqueConstraint("business_id", "workspace_id", name="uq_config_workspace"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("businesses.id"))
    workspace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workspaces.id"))

    chunk_words: Mapped[int] = mapped_column(Integer, default=300)
    overlap_words: Mapped[int] = mapped_column(Integer, default=50)
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.2)
    max_context_chars: Mapped[int] = mapped_column(Integer, default=12000)
    embedding_model: Mapped[str] = mapped_column(String(255), default="text-embedding-3-small")
    chat_model_default: Mapped[str] = mapped_column(String(255), default="gpt-4.1-mini")
    chat_temperature_default: Mapped[float] = mapped_column(Float, default=0.2)
    chat_max_tokens_default: Mapped[int] = mapped_column(Integer, default=600)

    workspace = relationship("Workspace", back_populates="config")
