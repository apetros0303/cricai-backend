import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from api.limiter import limiter
from api.routes import matches, predictions
from services.cricapi_client import CricApiClient
from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CricAI backend starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    if not settings.CRICAPI_KEY:
        logger.warning("CRICAPI_KEY not set — cricket data endpoints will fail")
    if not settings.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — AI analysis will be unavailable")
    if not settings.REVENUECAT_API_KEY:
        logger.warning("REVENUECAT_API_KEY not set — all premium requests will be denied")
    yield
    logger.info("CricAI backend shutting down.")


app = FastAPI(
    title="CricAI",
    version="1.0.0",
    lifespan=lifespan,
    # Docs disabled in production — exposes endpoint/param surface to scrapers
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(matches.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "api_requests_today": CricApiClient.requests_used_today(),
        "daily_quota": settings.CRICAPI_DAILY_LIMIT,
        "supported_languages": ["en", "hi", "ur"],
    }


@app.get("/")
async def root():
    return {"message": "CricAI API"}
