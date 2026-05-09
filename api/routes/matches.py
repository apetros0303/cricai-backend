import logging
from fastapi import APIRouter, Query, HTTPException
from models.match import CricketMatch
from services.cricket_service import CricketService
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/matches", tags=["Matches"])
settings = get_settings()


@router.get("/live", response_model=list[CricketMatch])
async def get_live_matches():
    """All currently live cricket matches."""
    svc = CricketService()
    return await svc.get_live_matches()


@router.get("/upcoming", response_model=list[CricketMatch])
async def get_upcoming_matches(
    series_id: str | None = Query(default=None, description="Filter by series ID"),
):
    """Upcoming matches, optionally filtered by series."""
    svc = CricketService()
    return await svc.get_upcoming_matches(series_id=series_id)


@router.get("/series/{series_id}", response_model=list[CricketMatch])
async def get_series_matches(series_id: str):
    """All matches in a series (e.g. IPL 2025)."""
    svc = CricketService()
    matches = await svc.get_series_matches(series_id)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No matches found for series {series_id}")
    return matches


@router.get("/series", response_model=dict)
async def list_known_series():
    """Returns the configured series IDs (IPL, PSL, etc.)."""
    return settings.CRICKET_SERIES


@router.get("/{match_id}", response_model=CricketMatch)
async def get_match(match_id: str):
    """Single match details."""
    svc = CricketService()
    match = await svc.get_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
    return match
