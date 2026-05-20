import logging
from typing import Optional
from fastapi import APIRouter, Query, HTTPException, Header, Request
from models.prediction import CricketPrediction
from models.match import TeamBattingForm, CricketFormat
from services.cricket_service import CricketService
from services.cricket_predictor import get_engine
from services.cricket_ai_analyst import CricketAiAnalyst
from services.revenuecat import is_premium_user
from api.limiter import limiter
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["Predictions"])
settings = get_settings()

_ai_analyst = CricketAiAnalyst()


@router.get("/{match_id}", response_model=CricketPrediction)
@limiter.limit("20/minute")
async def predict_match(
    request: Request,
    match_id: str,
    language: str = Query(default="en", description="Language for AI analysis: en, hi, ur"),
    x_rc_user_id: Optional[str] = Header(default=None),
):
    """
    Full prediction for a cricket match.
    - Free: match winner probabilities, total runs O/U, form, H2H
    - Premium (verified via X-RC-User-ID header): player predictions, AI analysis, value bet
    """
    if language not in ("en", "hi", "ur"):
        raise HTTPException(status_code=400, detail="language must be one of: en, hi, ur")

    premium = await is_premium_user(x_rc_user_id)

    svc = CricketService()

    match = await svc.get_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    # Fetch recent matches for form/H2H derivation.
    # get_upcoming_matches only returns non-finished matches, so we also pull the
    # raw paginated feed (offset 0 + 25) which includes recently-finished games.
    recent_all = await svc.get_upcoming_matches()
    try:
        recent_all += await svc.get_recent_finished_matches()
    except Exception as e:
        logger.warning(f"Recent finished matches fetch failed: {e}")

    team1_form = await svc.get_team_form(match.team1.name, recent_all)
    team2_form = await svc.get_team_form(match.team2.name, recent_all)
    h2h = await svc.get_head_to_head(match.team1.name, match.team2.name, recent_all)

    if team1_form.matches_played == 0:
        team1_form = TeamBattingForm(team_id=match.team1.id, team_name=match.team1.name)
    if team2_form.matches_played == 0:
        team2_form = TeamBattingForm(team_id=match.team2.id, team_name=match.team2.name)

    team1_squad, team2_squad = [], []
    if premium:
        try:
            team1_squad, team2_squad = await svc.get_squad_form(match.id)
        except Exception as e:
            logger.warning(f"Squad fetch failed for {match_id}: {e}")

    engine = get_engine(match.format)

    common = dict(
        match_id=match.id,
        team1=match.team1.name,
        team2=match.team2.name,
        team1_logo=match.team1.logo_url,
        team2_logo=match.team2.logo_url,
        match_start_utc=match.match_start_utc,
        series_name=match.series_name,
        team1_form=team1_form,
        team2_form=team2_form,
        h2h=h2h,
        venue=match.venue,
        team1_squad=team1_squad,
        team2_squad=team2_squad,
    )

    if match.format == CricketFormat.TEST:
        prediction = engine.predict(**common)
    else:
        prediction = engine.predict(
            **common,
            toss_winner=match.toss_winner,
            toss_decision=match.toss_decision,
        )

    prediction.language = language

    if premium:
        try:
            analysis, factors, ai_conf = await _ai_analyst.analyze_match(
                team1=match.team1.name,
                team2=match.team2.name,
                series=match.series_name,
                fmt=match.format,
                team1_form=team1_form,
                team2_form=team2_form,
                h2h=h2h,
                prediction=prediction,
                language=language,
                venue_notes=(
                    "Dew factor expected" if match.venue and match.venue.dew_factor else ""
                ),
            )
            prediction.ai_analysis = analysis
            prediction.ai_key_factors = factors
            prediction.ai_confidence = ai_conf
        except Exception as e:
            logger.error(f"AI analysis failed for match {match_id}: {e}")

    return prediction


@router.get("/bulk/series/{series_id}", response_model=list[CricketPrediction])
@limiter.limit("10/minute")
async def predict_series_bulk(
    request: Request,
    series_id: str,
    language: str = Query(default="en"),
    x_rc_user_id: Optional[str] = Header(default=None),
):
    """Predictions for all upcoming matches in a series."""
    if language not in ("en", "hi", "ur"):
        raise HTTPException(status_code=400, detail="language must be one of: en, hi, ur")

    premium = await is_premium_user(x_rc_user_id)

    svc = CricketService()
    matches = await svc.get_series_matches(series_id)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No matches found for series {series_id}")

    recent_all = await svc.get_upcoming_matches()
    results: list[CricketPrediction] = []

    for match in matches:
        try:
            team1_form = await svc.get_team_form(match.team1.name, recent_all)
            team2_form = await svc.get_team_form(match.team2.name, recent_all)
            h2h = await svc.get_head_to_head(match.team1.name, match.team2.name, recent_all)

            if team1_form.matches_played == 0:
                team1_form = TeamBattingForm(team_id=match.team1.id, team_name=match.team1.name)
            if team2_form.matches_played == 0:
                team2_form = TeamBattingForm(team_id=match.team2.id, team_name=match.team2.name)

            engine = get_engine(match.format)
            common = dict(
                match_id=match.id,
                team1=match.team1.name,
                team2=match.team2.name,
                team1_logo=match.team1.logo_url,
                team2_logo=match.team2.logo_url,
                match_start_utc=match.match_start_utc,
                series_name=match.series_name,
                team1_form=team1_form,
                team2_form=team2_form,
                h2h=h2h,
                venue=match.venue,
            )

            if match.format == CricketFormat.TEST:
                pred = engine.predict(**common)
            else:
                pred = engine.predict(**common, toss_winner=match.toss_winner, toss_decision=match.toss_decision)

            pred.language = language

            if premium:
                try:
                    analysis, factors, ai_conf = await _ai_analyst.analyze_match(
                        team1=match.team1.name,
                        team2=match.team2.name,
                        series=match.series_name,
                        fmt=match.format,
                        team1_form=team1_form,
                        team2_form=team2_form,
                        h2h=h2h,
                        prediction=pred,
                        language=language,
                    )
                    pred.ai_analysis = analysis
                    pred.ai_key_factors = factors
                    pred.ai_confidence = ai_conf
                except Exception:
                    pass

            results.append(pred)
        except Exception as e:
            logger.warning(f"Skipping match {match.id}: {e}")
            continue

    return results
