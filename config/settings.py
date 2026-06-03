from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    CRICAPI_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    SECRET_KEY: str = "change_me"
    ENVIRONMENT: str = "development"

    # RevenueCat server-side verification — set to the Secret API Key from RC dashboard
    REVENUECAT_API_KEY: str = ""
    REVENUECAT_ENTITLEMENT_ID: str = "Premium"

    CRICAPI_BASE_URL: str = "https://api.cricapi.com/v1"

    # Paid plan: 10000/day. Free tier: 100/day.
    CRICAPI_DAILY_LIMIT: int = 10000

    FREE_AI_TIPS_PER_DAY: int = 3

    CURRENT_SEASON: int = 2026

    # Series IDs on CricAPI (update each season)
    CRICKET_SERIES: dict = {
        "India vs NZ": "9f0a2527-1508-42b3-9945-4ade13fd4b5a",       # India tour of New Zealand 2026
        "ODI Qualifier": "7ac098a3-5d70-4c81-8e81-199edbeffc29",      # ICC WC League Two 2023-27
        "ICC Womens T20 WC": "f3e5c7dd-332c-4893-9067-aa2bfe6d2b85", # ICC Women's T20 WC 2026
        "CPL": "929c36f6-9ed6-4cec-a2ad-910a2ee4f701",                # Caribbean Premier League 2026
        "IPL": "87c62aac-bc3c-4738-ab93-19da0690488f",                # Indian Premier League 2026
    }

    # Prediction engine weights
    BATTING_STRENGTH_WEIGHT: float = 0.55
    BOWLING_STRENGTH_WEIGHT: float = 0.45
    VENUE_FACTOR_WEIGHT: float = 0.10
    H2H_BLEND_WEIGHT: float = 0.15
    FORM_MATCHES: int = 5

    # Value bet threshold
    VALUE_BET_THRESHOLD: float = 0.60


@lru_cache
def get_settings() -> Settings:
    return Settings()
