from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db.session import get_session
from app.models import Business, BusinessAdmin, Workspace, WorkspaceConfig
from app.schemas.workspace import WorkspaceCreate, WorkspaceOut

router = APIRouter(
    prefix="/admin/businesses/{business_client_id}/workspaces",
    tags=["Workspaces"],
)


async def _get_business(session: AsyncSession, business_client_id: str) -> Business:
    stmt = select(Business).where(Business.business_client_id == business_client_id)
    business = (await session.execute(stmt)).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    return business


def _ensure_access(admin: BusinessAdmin, business: Business) -> None:
    if admin.role != "super_admin" and admin.business_id != business.id:
        raise HTTPException(status_code=403, detail="Not allowed")


@router.post("", response_model=WorkspaceOut)
async def create_workspace(
    business_client_id: str,
    payload: WorkspaceCreate,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> WorkspaceOut:
    business = await _get_business(session, business_client_id)
    _ensure_access(admin, business)
    existing = (
        await session.execute(
            select(Workspace).where(
                Workspace.business_id == business.id,
                Workspace.workspace_id == payload.workspace_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Workspace already exists")

    workspace = Workspace(
        business_id=business.id, workspace_id=payload.workspace_id, name=payload.name
    )
    session.add(workspace)
    await session.flush()
    config = WorkspaceConfig(business_id=business.id, workspace_id=workspace.id)
    session.add(config)
    await session.commit()
    await session.refresh(workspace)
    return workspace


@router.get("", response_model=list[WorkspaceOut])
async def list_workspaces(
    business_client_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> list[WorkspaceOut]:
    business = await _get_business(session, business_client_id)
    _ensure_access(admin, business)
    stmt = select(Workspace).where(Workspace.business_id == business.id)
    return list((await session.execute(stmt)).scalars().all())


@router.get("/{workspace_id}", response_model=WorkspaceOut)
async def get_workspace(
    business_client_id: str,
    workspace_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> WorkspaceOut:
    business = await _get_business(session, business_client_id)
    _ensure_access(admin, business)
    stmt = select(Workspace).where(
        Workspace.business_id == business.id, Workspace.workspace_id == workspace_id
    )
    workspace = (await session.execute(stmt)).scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@router.delete("/{workspace_id}")
async def delete_workspace(
    business_client_id: str,
    workspace_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> dict:
    business = await _get_business(session, business_client_id)
    _ensure_access(admin, business)
    stmt = select(Workspace).where(
        Workspace.business_id == business.id, Workspace.workspace_id == workspace_id
    )
    workspace = (await session.execute(stmt)).scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    await session.delete(workspace)
    await session.commit()
    return {"status": "deleted", "cascade": "documents_and_chunks"}
