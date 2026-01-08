import uuid

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Workspace(Base):
    __tablename__ = "workspaces"
    __table_args__ = (
        UniqueConstraint("business_id", "workspace_id", name="uq_workspace_business"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("businesses.id"))
    workspace_id: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(255))

    business = relationship("Business", back_populates="workspaces")
    config = relationship("WorkspaceConfig", back_populates="workspace", uselist=False)
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")
