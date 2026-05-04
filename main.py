import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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
    yield
    logger.info("CricAI backend shutting down.")


app = FastAPI(
    title="CricAI",
    description="AI-powered cricket predictions API — English, Hindi, Urdu",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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


@app.get("/debug/raw")
async def debug_raw():
    import httpx, os
    key_from_settings = settings.CRICAPI_KEY
    key_from_env = os.environ.get("CRICAPI_KEY", "NOT_FOUND")
    key = key_from_env if key_from_env != "NOT_FOUND" else key_from_settings
    results = {
        "key_from_settings_len": len(key_from_settings),
        "key_from_env_len": len(key_from_env),
        "key_from_env_preview": f"{key_from_env[:8]}...{key_from_env[-4:]}" if len(key_from_env) > 8 else key_from_env,
        "key_used_len": len(key),
    }
    url = "https://api.cricapi.com/v1/currentMatches"
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.get(url, params={"apikey": key, "offset": 0})
        results["api"] = {"status_code": r.status_code, "body": r.json()}
    except Exception as e:
        results["api"] = {"error": repr(e)}
    return results


@app.get("/")
async def root():
    return {"message": "CricAI API", "docs": "/docs"}
