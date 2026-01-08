"""initial schema

Revision ID: 0001_initial
Revises: 
Create Date: 2025-02-14 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "businesses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_client_id", sa.String(length=100), unique=True, index=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id")),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("business_id", "workspace_id", name="uq_workspace_business"),
    )

    op.create_table(
        "business_admins",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id")),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("business_id", "email", name="uq_admin_business_email"),
    )

    op.create_table(
        "workspace_config",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id")),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id")),
        sa.Column("chunk_words", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("overlap_words", sa.Integer(), nullable=False, server_default="50"),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("similarity_threshold", sa.Float(), nullable=False, server_default="0.2"),
        sa.Column("max_context_chars", sa.Integer(), nullable=False, server_default="12000"),
        sa.Column(
            "embedding_model",
            sa.String(length=255),
            nullable=False,
            server_default="text-embedding-3-small",
        ),
        sa.Column(
            "chat_model_default",
            sa.String(length=255),
            nullable=False,
            server_default="gpt-4.1-mini",
        ),
        sa.Column("chat_temperature_default", sa.Float(), nullable=False, server_default="0.2"),
        sa.Column("chat_max_tokens_default", sa.Integer(), nullable=False, server_default="600"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("business_id", "workspace_id", name="uq_config_workspace"),
    )

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id")),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id")),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=50), nullable=False),
        sa.Column("storage_path", sa.String(length=500), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("meta_json", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id")),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id")),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_chunk_business_workspace", "document_chunks", ["business_id", "workspace_id"])
    op.create_index(
        "ix_chunk_business_workspace_doc",
        "document_chunks",
        ["business_id", "workspace_id", "document_id"],
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "chat_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("businesses.id")),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id")),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("query_text", sa.String(), nullable=False),
        sa.Column("retrieved_chunk_ids", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "chat_responses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_requests.id")),
        sa.Column("answer_text", sa.String(), nullable=False),
        sa.Column("sources_json", sa.String(), nullable=False),
        sa.Column("model_used", sa.String(length=255), nullable=False),
        sa.Column("tokens_json", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("chat_responses")
    op.drop_table("chat_requests")
    op.drop_index("ix_chunk_business_workspace_doc", table_name="document_chunks")
    op.drop_index("ix_chunk_business_workspace", table_name="document_chunks")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("workspace_config")
    op.drop_table("business_admins")
    op.drop_table("workspaces")
    op.drop_table("businesses")
