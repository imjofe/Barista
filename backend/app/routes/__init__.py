"""API router configuration for the FastAPI application."""

from fastapi import APIRouter

from . import chat, health, voice

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(voice.router)

__all__ = ["api_router"]

