"""add system_config table for OpenAI API key etc.

Revision ID: 0004_system_config
Revises: 0003_vec384
Create Date: 2026-02-06 12:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_system_config"
down_revision = "0003_vec384"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_config",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value", sa.String(length=2000), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_system_config_key", "system_config", ["key"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_system_config_key", table_name="system_config")
    op.drop_table("system_config")
