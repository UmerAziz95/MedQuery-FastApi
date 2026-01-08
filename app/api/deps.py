import uuid

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.db.session import get_session
from app.models import BusinessAdmin

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/auth/login")
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/auth/login", auto_error=False)


async def get_current_admin(
    session: AsyncSession = Depends(get_session), token: str = Depends(oauth2_scheme)
) -> BusinessAdmin:
    try:
        payload = decode_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    stmt = select(BusinessAdmin).where(BusinessAdmin.id == uuid.UUID(payload.sub))
    result = await session.execute(stmt)
    admin = result.scalar_one_or_none()
    if not admin:
        raise HTTPException(status_code=401, detail="Admin not found")
    return admin


def require_super_admin(admin: BusinessAdmin = Depends(get_current_admin)) -> BusinessAdmin:
    if admin.role != "super_admin":
        raise HTTPException(status_code=403, detail="Super admin required")
    return admin
