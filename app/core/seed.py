import logging
import uuid

from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError, OperationalError

from app.core.security import get_password_hash
from app.db.session import AsyncSessionLocal
from app.models import Business, BusinessAdmin, Workspace, WorkspaceConfig

logger = logging.getLogger(__name__)

DEFAULT_BUSINESS_CLIENT_ID = "default"
DEFAULT_BUSINESS_NAME = "Default Business"
DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_PASSWORD = "password"
DEFAULT_WORKSPACE_ID = "main"
DEFAULT_WORKSPACE_NAME = "Main Workspace"


async def seed_initial_admin() -> None:
    """Seed initial admin user. Gracefully handles missing database tables."""
    try:
        async with AsyncSessionLocal() as session:
            # ✅ Allow multiple admins in DB:
            # Only skip seeding if the DEFAULT admin email already exists.
            try:
                existing_admin_id = (
                    await session.execute(
                        select(BusinessAdmin.id).where(BusinessAdmin.email == DEFAULT_ADMIN_EMAIL)
                    )
                ).scalar_one_or_none()
            except (ProgrammingError, OperationalError) as e:
                # Database tables don't exist yet - migrations need to be run
                error_msg = str(e).lower()
                if "does not exist" in error_msg or "relation" in error_msg:
                    logger.warning(
                        "Database tables not found. Skipping seed. "
                        "Please run migrations first: 'docker compose exec api alembic upgrade head'"
                    )
                    return
                # Re-raise if it's a different database error
                raise

            if existing_admin_id:
                logger.info(f"Default admin ({DEFAULT_ADMIN_EMAIL}) already exists, skipping seed")
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
            logger.info(
                f"Seeded default admin: {DEFAULT_ADMIN_EMAIL} / {DEFAULT_ADMIN_PASSWORD} "
                f"(business: {DEFAULT_BUSINESS_CLIENT_ID}, workspace: {DEFAULT_WORKSPACE_ID})"
            )
    except Exception as e:
        logger.error(f"Error during seed_initial_admin: {type(e).__name__}: {e}", exc_info=True)
        # Don't crash the app if seeding fails - just log the error
        # This allows the app to start even if seeding fails
