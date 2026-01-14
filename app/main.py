from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import (
    auth_router,
    businesses_router,
    chat_router,
    documents_router,
    health_router,
    rag_router,
    ui_router,
    workspace_config_router,
    workspaces_router,
)
from app.core.config import get_settings
from app.core.seed import seed_initial_admin

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await seed_initial_admin()
    yield


app = FastAPI(
    title=settings.app_name,
    openapi_url="/openapi.json",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": exc.status_code, "message": exc.detail, "details": {}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"status": 500, "message": "Internal server error", "details": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"status": 422, "message": "Validation error", "details": exc.errors()},
    )


app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(businesses_router, prefix=settings.api_v1_prefix)
app.include_router(workspaces_router, prefix=settings.api_v1_prefix)
app.include_router(workspace_config_router, prefix=settings.api_v1_prefix)
app.include_router(documents_router, prefix=settings.api_v1_prefix)
app.include_router(rag_router, prefix=settings.api_v1_prefix)
app.include_router(chat_router, prefix=settings.api_v1_prefix)
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(ui_router)
