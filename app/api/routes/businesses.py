from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db.session import get_session
from app.models import Business, BusinessAdmin
from app.schemas.business import BusinessCreate, BusinessOut

router = APIRouter(prefix="/admin/businesses", tags=["Businesses"])


def _ensure_super_admin(admin: BusinessAdmin) -> None:
    if admin.role != "super_admin":
        raise HTTPException(status_code=403, detail="Super admin required")


@router.post("", response_model=BusinessOut)
async def create_business(
    payload: BusinessCreate,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> BusinessOut:
    _ensure_super_admin(admin)
    existing = (
        await session.execute(
            select(Business).where(Business.business_client_id == payload.business_client_id)
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Business already exists")

    business = Business(business_client_id=payload.business_client_id, name=payload.name)
    session.add(business)
    await session.commit()
    await session.refresh(business)
    return business


@router.get("", response_model=list[BusinessOut])
async def list_businesses(
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> list[BusinessOut]:
    if admin.role != "super_admin":
        stmt = select(Business).where(Business.id == admin.business_id)
        return list((await session.execute(stmt)).scalars().all())

    stmt = select(Business)
    return list((await session.execute(stmt)).scalars().all())


@router.get("/{business_client_id}", response_model=BusinessOut)
async def get_business(
    business_client_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> BusinessOut:
    stmt = select(Business).where(Business.business_client_id == business_client_id)
    business = (await session.execute(stmt)).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    if admin.role != "super_admin" and business.id != admin.business_id:
        raise HTTPException(status_code=403, detail="Not allowed")
    return business
