import logging
import httpx
from cachetools import TTLCache
from config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# Cache verification results for 1 hour — avoids hammering RevenueCat on every request
_cache: TTLCache[str, bool] = TTLCache(maxsize=2000, ttl=3600)


async def is_premium_user(app_user_id: str | None) -> bool:
    """
    Verify a RevenueCat app_user_id has an active premium entitlement.
    Returns False if the key is unconfigured, the user ID is blank, or the
    RevenueCat call fails — so the system fails closed on errors.
    """
    if not app_user_id or not app_user_id.strip():
        return False
    if not _settings.REVENUECAT_API_KEY:
        logger.warning("REVENUECAT_API_KEY not set — all premium requests denied")
        return False

    uid = app_user_id.strip()
    if uid in _cache:
        return _cache[uid]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"https://api.revenuecat.com/v1/subscribers/{uid}",
                headers={"Authorization": f"Bearer {_settings.REVENUECAT_API_KEY}"},
            )
        active = (
            resp.status_code == 200
            and _settings.REVENUECAT_ENTITLEMENT_ID
            in resp.json().get("subscriber", {}).get("entitlements", {})
        )
    except Exception as exc:
        logger.error(f"RevenueCat verification failed for {uid}: {exc}")
        active = False

    _cache[uid] = active
    return active
