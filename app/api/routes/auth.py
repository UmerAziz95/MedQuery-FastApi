import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import optional_oauth2_scheme
from app.core.security import create_access_token, decode_token, get_password_hash, verify_password
from app.db.session import get_session
from app.models import Business, BusinessAdmin
from app.schemas.auth import CreateAdminRequest, LoginRequest, TokenResponse

router = APIRouter(prefix="/admin/auth", tags=["Admin Auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, session: AsyncSession = Depends(get_session)) -> TokenResponse:
    stmt = select(Business).where(Business.business_client_id == request.business_client_id)
    business = (await session.execute(stmt)).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    stmt = select(BusinessAdmin).where(
        BusinessAdmin.business_id == business.id, BusinessAdmin.email == request.email
    )
    admin = (await session.execute(stmt)).scalar_one_or_none()
    if not admin or not verify_password(request.password, admin.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(admin.id), str(business.id), admin.role)
    return TokenResponse(access_token=token)


@router.post("/create-admin")
async def create_admin(
    request: CreateAdminRequest,
    session: AsyncSession = Depends(get_session),
    token: str | None = Depends(optional_oauth2_scheme),
) -> dict:
    admins_exist = (await session.execute(select(BusinessAdmin))).scalar_one_or_none()
    if admins_exist:
        if not token:
            raise HTTPException(status_code=403, detail="Super admin required")
        payload = decode_token(token)
        stmt = select(BusinessAdmin).where(BusinessAdmin.id == uuid.UUID(payload.sub))
        admin = (await session.execute(stmt)).scalar_one_or_none()
        if not admin or admin.role != "super_admin":
            raise HTTPException(status_code=403, detail="Super admin required")
    stmt = select(Business).where(Business.business_client_id == request.business_client_id)
    business = (await session.execute(stmt)).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    stmt = select(BusinessAdmin).where(
        BusinessAdmin.business_id == business.id, BusinessAdmin.email == request.email
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Admin already exists")

    admin = BusinessAdmin(
        id=uuid.uuid4(),
        business_id=business.id,
        email=request.email,
        password_hash=get_password_hash(request.password),
        role=request.role,
    )
    session.add(admin)
    await session.commit()
    return {"status": "created"}
