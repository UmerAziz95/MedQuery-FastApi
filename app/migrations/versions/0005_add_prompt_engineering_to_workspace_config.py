"""add prompt_engineering to workspace_config

Revision ID: 0005_prompt_engineering
Revises: 0004_system_config
Create Date: 2026-02-06 14:00:00

"""

from alembic import op
import sqlalchemy as sa


revision = "0005_prompt_engineering"
down_revision = "0004_system_config"
branch_labels = None
depends_on = None


DEFAULT_PROMPT = "You are a medical assistant. Provide concise answers based on the context."


def upgrade() -> None:
    op.add_column(
        "workspace_config",
        sa.Column(
            "prompt_engineering",
            sa.Text(),
            nullable=False,
            server_default=DEFAULT_PROMPT,
        ),
    )


def downgrade() -> None:
    op.drop_column("workspace_config", "prompt_engineering")
