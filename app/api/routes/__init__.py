from app.api.routes.auth import router as auth_router
from app.api.routes.businesses import router as businesses_router
from app.api.routes.workspaces import router as workspaces_router
from app.api.routes.workspace_config import router as workspace_config_router
from app.api.routes.documents import router as documents_router
from app.api.routes.rag import router as rag_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.ui import router as ui_router

__all__ = [
    "auth_router",
    "businesses_router",
    "workspaces_router",
    "workspace_config_router",
    "documents_router",
    "rag_router",
    "chat_router",
    "health_router",
    "ui_router",
]
