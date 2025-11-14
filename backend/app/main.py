"""FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import AppContext, get_settings
from app.routes import api_router

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="Barista Agent API",
        description="Backend service orchestrating the Barista AI agent.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.ctx = AppContext(settings=settings)

    Path(settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)

    @app.on_event("startup")
    async def on_startup() -> None:
        from agent.graph import build_agent_graph

        logger.info("startup.initializing_agent")

        app.state.ctx.agent_graph = build_agent_graph(app.state.ctx)
        logger.info("startup.complete", chroma_path=settings.chroma_persist_path)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("shutdown.complete")

    @app.get("/", summary="Root status")
    async def root() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router)

    return app


app = create_app()

