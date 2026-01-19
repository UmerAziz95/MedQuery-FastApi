from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_UI_PATH = Path(__file__).resolve().parents[2] / "ui" / "index.html"
_CHAT_PATH = Path(__file__).resolve().parents[2] / "ui" / "chat.html"
_DBVIEW_PATH = Path(__file__).resolve().parents[2] / "ui" / "dbview.html"
_ASSIGNMENTS_PATH = Path(__file__).resolve().parents[2] / "ui" / "assignments.html"
_PROFILE_PATH = Path(__file__).resolve().parents[2] / "ui" / "profile.html"
_SYSTEM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "ui" / "system-config.html"
_RAG_PATH = Path(__file__).resolve().parents[2] / "ui" / "rag.html"
_SYSTEM_LOGS_PATH = Path(__file__).resolve().parents[2] / "ui" / "system-logs.html"


@router.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    html_content = _UI_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/chat", response_class=HTMLResponse)
async def chat() -> HTMLResponse:
    html_content = _CHAT_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/dbview", response_class=HTMLResponse)
async def dbview() -> HTMLResponse:
    html_content = _DBVIEW_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/database-overview", response_class=HTMLResponse)
async def database_overview() -> HTMLResponse:
    html_content = _DBVIEW_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/assignments", response_class=HTMLResponse)
async def assignments() -> HTMLResponse:
    html_content = _ASSIGNMENTS_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/profile", response_class=HTMLResponse)
async def profile() -> HTMLResponse:
    html_content = _PROFILE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/system-config", response_class=HTMLResponse)
async def system_config() -> HTMLResponse:
    html_content = _SYSTEM_CONFIG_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/rag-builder", response_class=HTMLResponse)
async def rag_builder() -> HTMLResponse:
    html_content = _RAG_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/system-logs", response_class=HTMLResponse)
async def system_logs() -> HTMLResponse:
    html_content = _SYSTEM_LOGS_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)
