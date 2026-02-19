from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SystemConfig

OPENAI_API_KEY_KEY = "openai_api_key"


async def get_openai_api_key(session: AsyncSession) -> str | None:
    """Return OpenAI API key from DB, or None if not set."""
    stmt = select(SystemConfig).where(SystemConfig.key == OPENAI_API_KEY_KEY)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if not row or not (row.value and row.value.strip()):
        return None
    return row.value.strip()


async def set_openai_api_key(session: AsyncSession, value: str) -> None:
    """Set OpenAI API key in DB. Creates or updates the row."""
    stmt = select(SystemConfig).where(SystemConfig.key == OPENAI_API_KEY_KEY)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row:
        row.value = value.strip()
    else:
        session.add(SystemConfig(key=OPENAI_API_KEY_KEY, value=value.strip()))
    await session.commit()


async def get_openai_api_key_status(session: AsyncSession) -> dict:
    """Return { set: bool, masked_key: str } for UI (never the full key)."""
    key = await get_openai_api_key(session)
    if not key:
        return {"set": False, "masked_key": None}
    if len(key) <= 11:
        return {"set": True, "masked_key": "(set)"}
    return {"set": True, "masked_key": f"{key[:7]}...{key[-4:]}"}
