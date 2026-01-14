import uuid

from sqlalchemy import select

from app.core.security import get_password_hash
from app.db.session import AsyncSessionLocal
from app.models import Business, BusinessAdmin

DEFAULT_BUSINESS_CLIENT_ID = "default"
DEFAULT_BUSINESS_NAME = "Default Business"
DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_PASSWORD = "password"


async def seed_initial_admin() -> None:
    async with AsyncSessionLocal() as session:
        existing_admin = (await session.execute(select(BusinessAdmin))).scalar_one_or_none()
        if existing_admin:
            return

        business = (
            await session.execute(
                select(Business).where(Business.business_client_id == DEFAULT_BUSINESS_CLIENT_ID)
            )
        ).scalar_one_or_none()
        if not business:
            business = Business(
                business_client_id=DEFAULT_BUSINESS_CLIENT_ID, name=DEFAULT_BUSINESS_NAME
            )
            session.add(business)
            await session.flush()

        admin = BusinessAdmin(
            id=uuid.uuid4(),
            business_id=business.id,
            email=DEFAULT_ADMIN_EMAIL,
            password_hash=get_password_hash(DEFAULT_ADMIN_PASSWORD),
            role="super_admin",
        )
        session.add(admin)
        await session.commit()
