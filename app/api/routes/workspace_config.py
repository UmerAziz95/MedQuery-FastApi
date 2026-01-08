from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db.session import get_session
from app.models import Business, BusinessAdmin, Workspace, WorkspaceConfig
from app.schemas.workspace_config import WorkspaceConfigOut, WorkspaceConfigUpdate

router = APIRouter(
    prefix="/admin/businesses/{business_client_id}/workspaces/{workspace_id}/config",
    tags=["Workspace Config"],
)


async def _get_workspace(
    session: AsyncSession, business_client_id: str, workspace_id: str
) -> tuple[Business, Workspace]:
    business = (
        await session.execute(
            select(Business).where(Business.business_client_id == business_client_id)
        )
    ).scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    workspace = (
        await session.execute(
            select(Workspace).where(
                Workspace.business_id == business.id,
                Workspace.workspace_id == workspace_id,
            )
        )
    ).scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return business, workspace


def _ensure_access(admin: BusinessAdmin, business: Business) -> None:
    if admin.role != "super_admin" and admin.business_id != business.id:
        raise HTTPException(status_code=403, detail="Not allowed")


@router.get("", response_model=WorkspaceConfigOut)
async def get_workspace_config(
    business_client_id: str,
    workspace_id: str,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> WorkspaceConfigOut:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    stmt = select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
    config = (await session.execute(stmt)).scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config


@router.put("", response_model=WorkspaceConfigOut)
async def update_workspace_config(
    business_client_id: str,
    workspace_id: str,
    payload: WorkspaceConfigUpdate,
    session: AsyncSession = Depends(get_session),
    admin: BusinessAdmin = Depends(get_current_admin),
) -> WorkspaceConfigOut:
    business, workspace = await _get_workspace(session, business_client_id, workspace_id)
    _ensure_access(admin, business)
    stmt = select(WorkspaceConfig).where(WorkspaceConfig.workspace_id == workspace.id)
    config = (await session.execute(stmt)).scalar_one_or_none()
    if not config:
        config = WorkspaceConfig(business_id=business.id, workspace_id=workspace.id)
        session.add(config)
    for key, value in payload.model_dump().items():
        setattr(config, key, value)
    await session.commit()
    await session.refresh(config)
    return config
