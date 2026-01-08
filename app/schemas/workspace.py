from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    workspace_id: str = Field(..., example="main")
    name: str = Field(..., example="Main Workspace")


class WorkspaceOut(BaseModel):
    workspace_id: str
    name: str

    class Config:
        from_attributes = True
