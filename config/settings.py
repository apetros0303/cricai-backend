from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    CRICAPI_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    SECRET_KEY: str = "change_me"
    ENVIRONMENT: str = "development"

    CRICAPI_BASE_URL: str = "https://api.cricapi.com/v1"

    # Paid plan: 10000/day. Free tier: 100/day.
    CRICAPI_DAILY_LIMIT: int = 10000

    FREE_AI_TIPS_PER_DAY: int = 3

    CURRENT_SEASON: int = 2025

    # Series IDs on CricAPI (update each season)
    CRICKET_SERIES: dict = {
        # T20 leagues
        "IPL": "d5a498c8-7596-4b93-8ab0-e0efc3345312",
        "PSL": "3e1e7a33-b944-44c4-9beb-d0da55fa95bc",
        "Big Bash": "a5c5e6f8-1b34-4c78-8fe2-d1d34a5f6612",
        # International
        "ICC T20 World Cup": "d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a",
        "ICC ODI World Cup": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        # Test series
        "ICC World Test Championship": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
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
