"""
CricketData.org async client (same API interface as CricAPI).
Docs: https://cricketdata.org
Paid S plan: 2,000 hits/day ($5.99/mo). M plan: 10,000/day ($12.99/mo).
Key passed as query param `apikey`.
"""

import httpx
import logging
from cachetools import TTLCache
from typing import Any
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Shared in-memory caches — kept small to stay within 512MB Render instance
_match_cache: TTLCache = TTLCache(maxsize=60, ttl=3600)        # 1 h
_scorecard_cache: TTLCache = TTLCache(maxsize=80, ttl=604800)  # 7 days — finished scorecards never change
_series_cache: TTLCache = TTLCache(maxsize=30, ttl=86400)      # 24 h
_player_cache: TTLCache = TTLCache(maxsize=150, ttl=3600)      # 1 h
_live_cache: TTLCache = TTLCache(maxsize=20, ttl=60)           # 1 min for live data

_daily_requests: dict = {"count": 0, "date": ""}


class CricApiClient:
    """Async wrapper for CricAPI v1 endpoints."""

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.base_url = settings.CRICAPI_BASE_URL.rstrip("/")
        self.api_key = api_key or settings.CRICAPI_KEY
        self.daily_limit = settings.CRICAPI_DAILY_LIMIT

    def _track_request(self) -> None:
        from datetime import date
        today = str(date.today())
        if _daily_requests["date"] != today:
            _daily_requests["count"] = 0
            _daily_requests["date"] = today
        _daily_requests["count"] += 1
        remaining = self.daily_limit - _daily_requests["count"]
        if remaining <= 10:
            logger.warning(
                f"CricAPI quota warning: {_daily_requests['count']}/{self.daily_limit} used today. "
                f"{remaining} requests remaining."
            )

    async def _get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        self._track_request()
        all_params = {"apikey": self.api_key, **(params or {})}
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=all_params)
            response.raise_for_status()
            data = response.json()
        status = data.get("status", "")
        if status not in ("success", ""):
            raise ValueError(f"CricAPI error on {endpoint}: {data.get('reason', status)}")
        return data

    async def get_current_matches(self) -> list[dict]:
        """Live and recently started matches."""
        cache_key = "current_matches"
        if cache_key in _live_cache:
            return _live_cache[cache_key]
        data = await self._get("currentMatches", {"offset": 0})
        result = data.get("data", [])
        _live_cache[cache_key] = result
        return result

    async def get_matches(self, offset: int = 0) -> list[dict]:
        """Upcoming + recent matches (paginated, 25 per page)."""
        cache_key = f"matches:{offset}"
        if cache_key in _match_cache:
            return _match_cache[cache_key]
        data = await self._get("matches", {"offset": offset})
        result = data.get("data", [])
        _match_cache[cache_key] = result
        return result

    async def get_match_info(self, match_id: str) -> dict | None:
        """Full match details including scorecard."""
        cache_key = f"match:{match_id}"
        if cache_key in _match_cache:
            return _match_cache[cache_key]
        data = await self._get("match_info", {"id": match_id})
        result = data.get("data")
        if result:
            _match_cache[cache_key] = result
        return result

    async def get_match_scorecard(self, match_id: str) -> dict | None:
        """Detailed scorecard (innings breakdown). Cached for 7 days — finished scorecards are immutable."""
        cache_key = f"scorecard:{match_id}"
        if cache_key in _scorecard_cache:
            logger.debug(f"Scorecard cache hit: {match_id}")
            return _scorecard_cache[cache_key]
        data = await self._get("match_scorecard", {"id": match_id})
        result = data.get("data")
        if result:
            _scorecard_cache[cache_key] = result
        return result

    async def get_series(self, offset: int = 0) -> list[dict]:
        cache_key = f"series:{offset}"
        if cache_key in _series_cache:
            return _series_cache[cache_key]
        data = await self._get("series", {"offset": offset})
        result = data.get("data", [])
        _series_cache[cache_key] = result
        return result

    async def get_series_info(self, series_id: str) -> dict | None:
        cache_key = f"series_info:{series_id}"
        if cache_key in _series_cache:
            return _series_cache[cache_key]
        data = await self._get("series_info", {"id": series_id})
        result = data.get("data")
        if result:
            _series_cache[cache_key] = result
        return result

    async def get_series_matches(self, series_id: str) -> list[dict]:
        """All matches in a series."""
        cache_key = f"series_matches:{series_id}"
        if cache_key in _series_cache:
            return _series_cache[cache_key]
        info = await self.get_series_info(series_id)
        if not info:
            return []
        matches = info.get("matchList", [])
        _series_cache[cache_key] = matches
        return matches

    async def get_player_info(self, player_id: str) -> dict | None:
        cache_key = f"player:{player_id}"
        if cache_key in _player_cache:
            return _player_cache[cache_key]
        data = await self._get("players_info", {"id": player_id})
        result = data.get("data")
        if result:
            _player_cache[cache_key] = result
        return result

    async def search_players(self, name: str) -> list[dict]:
        data = await self._get("players", {"search": name, "offset": 0})
        return data.get("data", [])

    @staticmethod
    def requests_used_today() -> int:
        return _daily_requests.get("count", 0)
