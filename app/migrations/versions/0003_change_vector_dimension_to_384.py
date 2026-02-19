"""change vector dimension to 384 for local embeddings

Revision ID: 0003_vec384
Revises: 0002_add_use_local_embeddings
Create Date: 2025-02-02 10:10:00
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0003_vec384"
down_revision = "0002_add_use_local_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the existing index first
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")
    
    # Change the column type from vector(1536) to vector(384)
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)")
    
    # Recreate the index
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # Drop the index
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")
    
    # Change back to vector(1536)
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536)")
    
    # Recreate the index
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
