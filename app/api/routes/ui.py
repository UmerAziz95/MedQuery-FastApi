from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_UI_PATH = Path(__file__).resolve().parents[2] / "ui" / "index.html"
_DBVIEW_PATH = Path(__file__).resolve().parents[2] / "ui" / "dbview.html"
_ASSIGNMENTS_PATH = Path(__file__).resolve().parents[2] / "ui" / "assignments.html"


@router.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    html_content = _UI_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/dbview", response_class=HTMLResponse)
async def dbview() -> HTMLResponse:
    html_content = _DBVIEW_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/assignments", response_class=HTMLResponse)
async def assignments() -> HTMLResponse:
    html_content = _ASSIGNMENTS_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)
