import logging
import multiprocessing
import signal
import sys
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
    system_config_router,
    ui_router,
    workspace_config_router,
    workspaces_router,
)
from app.core.config import get_settings
from app.core.crash_logger import crash_logger
from app.core.seed import seed_initial_admin

settings = get_settings()

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle termination signals and log crash."""
    logger.critical(f"Received signal {signum}, logging crash before exit...")
    try:
        # Create a dummy exception for logging
        class SignalException(Exception):
            pass
        
        exc = SignalException(f"Application terminated by signal {signum}")
        crash_logger.log_crash(
            exc, type(exc), None,
            context={"signal_number": signum, "signal_name": signal.Signals(signum).name},
            additional_info={"frame_info": str(frame)}
        )
    except Exception as log_error:
        logger.error(f"Failed to log crash on signal: {log_error}")
    
    sys.exit(1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Register signal handlers for crash logging
    if sys.platform != "win32":  # Signal handlers work differently on Windows
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        # Note: SIGKILL cannot be caught
    
    logger.info("Application starting up...")
    # Use spawn for multiprocessing so PDF extraction subprocesses do not inherit parent memory (avoids 4GB spike)
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. by uvicorn)
    # Ensure crash/progress log dir exists (so ./logs/crashes appears on host volume)
    try:
        crash_logger.log_dir.mkdir(parents=True, exist_ok=True)
        crash_logger.write_progress("app_started", {"event": "startup"})
    except Exception as e:
        logger.warning(f"Could not init log dir: {e}")
    await seed_initial_admin()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down...")


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
    exc_type = type(exc)
    exc_message = str(exc)
    exc_traceback = exc.__traceback__
    
    # Determine if this is a critical crash
    is_critical = isinstance(exc, (MemoryError, SystemError, KeyboardInterrupt)) or \
                  "out of memory" in exc_message.lower() or \
                  "killed" in exc_message.lower()
    
    # Log with crash logger
    context = {
        "request_path": request.url.path,
        "request_method": request.method,
        "query_params": dict(request.query_params),
        "client_host": request.client.host if request.client else None,
    }
    
    if is_critical:
        crash_log_file = crash_logger.log_crash(
            exc, exc_type, exc_traceback,
            context=context,
            additional_info={
                "request_headers": dict(request.headers),
                "environment": settings.environment,
            }
        )
        logger.critical(f"CRITICAL CRASH LOGGED: {crash_log_file}")
    else:
        crash_logger.log_error(exc, exc_type, exc_traceback, context=context)
    
    exc_traceback_str = "".join(traceback.format_exception(exc_type, exc, exc_traceback))
    
    logger.error(
        f"Unhandled Exception: {exc_type.__name__} - {exc_message} | "
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Query Params: {dict(request.query_params)} | "
        f"Headers: {dict(request.headers)} | "
        f"Traceback:\n{exc_traceback_str}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "status": 500,
            "message": "Internal server error",
            "details": {
                "exception_type": exc_type.__name__,
                "exception_message": exc_message,
                "traceback": exc_traceback_str if settings.environment == "development" else None,
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
app.include_router(system_config_router, prefix=settings.api_v1_prefix)
app.include_router(dbview_router, prefix=settings.api_v1_prefix)
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(ui_router)
