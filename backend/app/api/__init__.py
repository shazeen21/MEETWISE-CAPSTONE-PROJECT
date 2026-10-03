"""API package for MeetWise AI."""

from .meetings import router as meetings_router
from .search import router as search_router

__all__ = ["meetings_router", "search_router"]

