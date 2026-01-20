import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import (
    auth_router,
    businesses_router,
    chat_router,
    documents_router,
    dbview_router,
    health_router,
    rag_router,
    ui_router,
    workspace_config_router,
    workspaces_router,
)
from app.core.config import get_settings
from app.core.seed import seed_initial_admin

settings = get_settings()

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        f"HTTPException: {exc.status_code} - {exc.detail} | "
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Headers: {dict(request.headers)}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": exc.status_code, "message": exc.detail, "details": {}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    exc_type = type(exc).__name__
    exc_message = str(exc)
    exc_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    
    logger.error(
        f"Unhandled Exception: {exc_type} - {exc_message} | "
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Query Params: {dict(request.query_params)} | "
        f"Headers: {dict(request.headers)} | "
        f"Traceback:\n{exc_traceback}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "status": 500,
            "message": "Internal server error",
            "details": {
                "exception_type": exc_type,
                "exception_message": exc_message,
                "traceback": exc_traceback if settings.environment == "development" else None,
            },
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    validation_errors = exc.errors()
    logger.warning(
        f"ValidationError: {len(validation_errors)} validation error(s) | "
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Query Params: {dict(request.query_params)} | "
        f"Errors: {validation_errors}"
    )
    return JSONResponse(
        status_code=422,
        content={"status": 422, "message": "Validation error", "details": validation_errors},
    )


app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(businesses_router, prefix=settings.api_v1_prefix)
app.include_router(workspaces_router, prefix=settings.api_v1_prefix)
app.include_router(workspace_config_router, prefix=settings.api_v1_prefix)
app.include_router(documents_router, prefix=settings.api_v1_prefix)
app.include_router(rag_router, prefix=settings.api_v1_prefix)
app.include_router(chat_router, prefix=settings.api_v1_prefix)
app.include_router(dbview_router, prefix=settings.api_v1_prefix)
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(ui_router)
