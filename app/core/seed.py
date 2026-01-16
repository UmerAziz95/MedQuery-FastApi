import uuid

from sqlalchemy import select

from app.core.security import get_password_hash
from app.db.session import AsyncSessionLocal
from app.models import Business, BusinessAdmin, Workspace, WorkspaceConfig

DEFAULT_BUSINESS_CLIENT_ID = "default"
DEFAULT_BUSINESS_NAME = "Default Business"
DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_PASSWORD = "password"
DEFAULT_WORKSPACE_ID = "main"
DEFAULT_WORKSPACE_NAME = "Main Workspace"


async def seed_initial_admin() -> None:
    async with AsyncSessionLocal() as session:
        # ✅ Allow multiple admins in DB:
        # Only skip seeding if the DEFAULT admin email already exists.
        existing_admin_id = (
            await session.execute(
                select(BusinessAdmin.id).where(BusinessAdmin.email == DEFAULT_ADMIN_EMAIL)
            )
        ).scalar_one_or_none()

        if existing_admin_id:
            return

        # Ensure default business exists
        business = (
            await session.execute(
                select(Business).where(
                    Business.business_client_id == DEFAULT_BUSINESS_CLIENT_ID
                )
            )
        ).scalar_one_or_none()

        if not business:
            business = Business(
                business_client_id=DEFAULT_BUSINESS_CLIENT_ID,
                name=DEFAULT_BUSINESS_NAME,
            )
            session.add(business)
            await session.flush()  # ensures business.id is available

        existing_workspace = (
            await session.execute(select(Workspace).where(Workspace.business_id == business.id))
        ).scalar_one_or_none()
        if not existing_workspace:
            workspace = Workspace(
                business_id=business.id,
                workspace_id=DEFAULT_WORKSPACE_ID,
                name=DEFAULT_WORKSPACE_NAME,
            )
            session.add(workspace)
            await session.flush()
            session.add(WorkspaceConfig(business_id=business.id, workspace_id=workspace.id))

        admin = BusinessAdmin(
            id=uuid.uuid4(),
            business_id=business.id,
            email=DEFAULT_ADMIN_EMAIL,
            password_hash=get_password_hash(DEFAULT_ADMIN_PASSWORD),
            role="super_admin",
        )
        session.add(admin)
        await session.commit()
