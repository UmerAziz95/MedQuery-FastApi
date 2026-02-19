"""add use_local_embeddings to workspace_config

Revision ID: 0002_add_use_local_embeddings
Revises: 0001_initial
Create Date: 2025-02-02 08:50:00
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_add_use_local_embeddings"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workspace_config",
        sa.Column("use_local_embeddings", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("workspace_config", "use_local_embeddings")
