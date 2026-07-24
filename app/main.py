from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.agent_chat import router as agent_chat_router
from app.api.ask import router as ask_router
from app.api.chunk import router as chunk_router
from app.api.embed import router as embed_router
from app.api.health import router as health_router
from app.api.process import router as process_router
from app.api.retrieve import router as retrieve_router
from app.api.upload import router as upload_router
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated.")
    for route in app.routes:
        methods = ",".join(sorted(route.methods)) if hasattr(route, "methods") else ""
        logger.info("Registered route: path=%s methods=%s", route.path, methods)
    yield
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade foundation for an Agentic AI Platform.",
    debug=settings.debug,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(process_router)
app.include_router(chunk_router)
app.include_router(embed_router)
app.include_router(retrieve_router)
app.include_router(ask_router)
app.include_router(agent_chat_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )
