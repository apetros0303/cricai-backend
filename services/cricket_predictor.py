"""
Cricket prediction engine.

Three separate models for each format:
  T20  — fast-scoring game, high variance, toss + dew factor matter a lot
  ODI  — balanced, powerplay phases weighted
  Test — draw is a real outcome; multi-day batting avg drives model

All models output a CricketPrediction with win probabilities, total runs O/U,
and optional player predictions.
"""

import numpy as np
from datetime import datetime
from models.match import (
    CricketFormat, TeamBattingForm, HeadToHead, VenueStats, PlayerForm
)
from models.prediction import CricketPrediction, PlayerPrediction
from config.settings import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# ICC team strength table (used when no real scorecard data available)
# Scores range 0.0 (weakest) to 1.0 (strongest). Strength 0.50 = benchmark.
# Based on ICC T20I/ODI rankings May 2025 + franchise team estimates.
# ---------------------------------------------------------------------------

_ICC_STRENGTH: dict[str, float] = {
    # Full ICC members — top tier
    "india": 0.90,
    "england": 0.82,
    "australia": 0.80,
    "south africa": 0.78,
    "pakistan": 0.76,
    "new zealand": 0.74,
    "west indies": 0.70,
    "afghanistan": 0.68,
    "sri lanka": 0.66,
    "bangladesh": 0.62,
    # Associate / emerging nations
    "zimbabwe": 0.54,
    "ireland": 0.53,
    "scotland": 0.51,
    "netherlands": 0.50,
    "namibia": 0.48,
    "nepal": 0.47,
    "oman": 0.46,
    "uae": 0.45,
    "united arab emirates": 0.45,
    "canada": 0.44,
    "usa": 0.44,
    "united states": 0.44,
    "kenya": 0.43,
    "hong kong": 0.45,
    "malaysia": 0.42,
    "indonesia": 0.38,
    "china": 0.38,
    "guernsey": 0.42,
    "isle of man": 0.41,
    "germany": 0.40,
    "austria": 0.39,
    "greece": 0.35,
    "cyprus": 0.34,
    # IPL franchises
    "mumbai indians": 0.72,
    "chennai super kings": 0.72,
    "kolkata knight riders": 0.70,
    "royal challengers bangalore": 0.68,
    "royal challengers bengaluru": 0.68,
    "sunrisers hyderabad": 0.67,
    "delhi capitals": 0.66,
    "rajasthan royals": 0.65,
    "punjab kings": 0.63,
    "lucknow super giants": 0.64,
    "gujarat titans": 0.64,
    # PSL franchises
    "karachi kings": 0.66,
    "lahore qalandars": 0.67,
    "peshawar zalmi": 0.64,
    "islamabad united": 0.65,
    "multan sultans": 0.65,
    "quetta gladiators": 0.63,
    # BBL franchises
    "sydney sixers": 0.64,
    "perth scorchers": 0.65,
    "melbourne stars": 0.62,
    "melbourne renegades": 0.60,
    "sydney thunder": 0.61,
    "hobart hurricanes": 0.62,
    "adelaide strikers": 0.63,
    "brisbane heat": 0.61,
}


def _team_strength(team_name: str) -> float:
    """Look up ICC/franchise strength index. Defaults to 0.50 for unknown teams."""
    name_lower = team_name.lower()
    for key, strength in _ICC_STRENGTH.items():
        if key in name_lower or name_lower in key:
            return strength
    return 0.50


def _synthetic_run_rates(team_name: str, bench: float) -> tuple[float, float]:
    """
    Generate synthetic avg_scored and avg_conceded from ICC rankings
    when real scorecard data is unavailable.
    Strength 0.5 maps to the venue benchmark. 1.0 = 25% above, 0.0 = 25% below.
    """
    strength = _team_strength(team_name)
    offset = 0.28 * (2 * strength - 1)  # -0.28 to +0.28
    avg_scored = bench * (1.0 + offset)
    avg_conceded = bench * (1.0 - offset)
    return avg_scored, avg_conceded


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.52:
        return "Medium"
    return "Low"


def _value_bet(probability: float) -> tuple[bool, float]:
    threshold = settings.VALUE_BET_THRESHOLD
    if probability >= threshold:
        fair_odds = 1.0 / probability
        offered_odds = fair_odds * 0.90   # assume 10% bookie margin
        ev = probability * offered_odds - 1.0
        return True, round(ev, 3)
    return False, 0.0


def _win_prob_from_run_rates(
    team1_avg_scored: float,
    team1_avg_conceded: float,
    team2_avg_scored: float,
    team2_avg_conceded: float,
    venue_first_innings: float,
) -> tuple[float, float]:
    """
    Derive win probabilities from run-scoring strength.
    Attack strength = avg scored / venue benchmark.
    Defense strength = avg conceded / venue benchmark (lower is better).
    Win prob via logistic on run differential estimate.
    """
    bench = venue_first_innings or 160.0

    t1_attack = team1_avg_scored / bench
    t1_defense = bench / max(team1_avg_conceded, 1)
    t2_attack = team2_avg_scored / bench
    t2_defense = bench / max(team2_avg_conceded, 1)

    # Composite strength index
    t1_strength = (t1_attack + t1_defense) / 2
    t2_strength = (t2_attack + t2_defense) / 2

    diff = t1_strength - t2_strength
    # logistic on difference (scale factor 4 gives reasonable spread)
    t1_win = 1.0 / (1.0 + np.exp(-diff * 4))
    t2_win = 1.0 - t1_win
    return float(t1_win), float(t2_win)


def _blend_h2h(t1_win: float, t2_win: float, h2h: HeadToHead) -> tuple[float, float]:
    if h2h.matches_played < 4:
        return t1_win, t2_win
    total_decisive = h2h.team1_wins + h2h.team2_wins
    if total_decisive == 0:
        return t1_win, t2_win
    h2h_t1 = h2h.team1_wins / total_decisive
    h2h_t2 = h2h.team2_wins / total_decisive
    w = settings.H2H_BLEND_WEIGHT
    blended_t1 = (1 - w) * t1_win + w * h2h_t1
    blended_t2 = (1 - w) * t2_win + w * h2h_t2
    # Renormalise
    total = blended_t1 + blended_t2
    return blended_t1 / total, blended_t2 / total


def _toss_adjustment(t1_win: float, t2_win: float, toss_winner: str,
                     toss_decision: str, team1_name: str, dew_factor: bool) -> tuple[float, float]:
    """
    Dew benefits chasing team (batting second). Winning toss and choosing to
    field/bowl in dew conditions gives a ~5% edge.
    """
    if not toss_winner or not toss_decision:
        return t1_win, t2_win
    boost = 0.05 if dew_factor else 0.02
    # If toss winner chose to field → they bat second → dew advantage
    chasing_team_advantage = toss_decision.lower() in ("field", "bowl")
    toss_winner_is_t1 = toss_winner.lower() in team1_name.lower()
    if chasing_team_advantage:
        if toss_winner_is_t1:
            t1_win = min(1.0, t1_win + boost)
        else:
            t2_win = min(1.0, t2_win + boost)
    total = t1_win + t2_win
    return t1_win / total, t2_win / total


def _top_player_predictions(players: list[PlayerForm], metric: str, n: int = 3) -> list[PlayerPrediction]:
    """Rank players by form score for bat/bowl and return top N with probabilities."""
    if not players:
        return []
    if metric == "bat":
        scored = sorted(players, key=lambda p: p.batting_form_score, reverse=True)
        scores = np.array([p.batting_form_score for p in scored])
    else:
        scored = sorted(players, key=lambda p: p.bowling_form_score, reverse=True)
        scores = np.array([p.bowling_form_score for p in scored])

    # Softmax to convert form scores into probabilities
    exp_s = np.exp(scores * 3)
    probs = exp_s / exp_s.sum()

    result = []
    for i, player in enumerate(scored[:n]):
        prob = float(probs[i])
        result.append(PlayerPrediction(
            player_id=player.player_id,
            player_name=player.player_name,
            probability=round(prob, 4),
            predicted_runs=round(player.batting_avg * player.batting_form_score * 1.1, 1) if metric == "bat" else None,
            predicted_wickets=round(sum(player.recent_wickets[:3]) / max(len(player.recent_wickets[:3]), 1), 1) if metric == "bowl" else None,
            form_score=player.batting_form_score if metric == "bat" else player.bowling_form_score,
            confidence_label=_confidence_label(prob * 2),  # scale up since it's a pick-one market
        ))
    return result


# ---------------------------------------------------------------------------
# T20 prediction engine
# ---------------------------------------------------------------------------

class T20PredictionEngine:
    """
    T20 model: run-rate strength × venue benchmark + form blend + H2H.
    Typical first innings: 150-190. High variance, toss + dew significant.
    """
    VENUE_BENCHMARK = 160.0

    def predict(
        self,
        match_id: str,
        team1: str,
        team2: str,
        team1_logo: str | None,
        team2_logo: str | None,
        match_start_utc: datetime,
        series_name: str,
        team1_form: TeamBattingForm,
        team2_form: TeamBattingForm,
        h2h: HeadToHead,
        venue: VenueStats | None = None,
        toss_winner: str | None = None,
        toss_decision: str | None = None,
        team1_squad: list[PlayerForm] | None = None,
        team2_squad: list[PlayerForm] | None = None,
    ) -> CricketPrediction:
        bench = (venue.avg_first_innings_score_t20 if venue else self.VENUE_BENCHMARK)
        dew = venue.dew_factor if venue else False

        # --- Win probabilities ---
        if team1_form.avg_runs_scored:
            t1_avg_scored = team1_form.avg_runs_scored
            t1_avg_conceded = team1_form.avg_runs_conceded or bench
        else:
            t1_avg_scored, t1_avg_conceded = _synthetic_run_rates(team1, bench)

        if team2_form.avg_runs_scored:
            t2_avg_scored = team2_form.avg_runs_scored
            t2_avg_conceded = team2_form.avg_runs_conceded or bench
        else:
            t2_avg_scored, t2_avg_conceded = _synthetic_run_rates(team2, bench)

        t1_win, t2_win = _win_prob_from_run_rates(
            t1_avg_scored, t1_avg_conceded, t2_avg_scored, t2_avg_conceded, bench
        )

        # --- Form blend (30% form, 70% strength) ---
        form_diff = team1_form.form_score - team2_form.form_score
        form_adj = form_diff * 0.10
        t1_win = max(0.05, min(0.95, t1_win + form_adj))
        t2_win = 1.0 - t1_win

        # --- H2H blend ---
        t1_win, t2_win = _blend_h2h(t1_win, t2_win, h2h)

        # --- Toss adjustment ---
        t1_win_post, t2_win_post = _toss_adjustment(t1_win, t2_win, toss_winner or "", toss_decision or "", team1, dew)

        # --- Total runs O/U ---
        avg_team_score = (t1_avg_scored + t2_avg_scored) / 2
        # Both teams bat once → total match runs ≈ first innings + second innings
        predicted_total = avg_team_score + (avg_team_score * 0.95)  # chasing team slightly less
        runs_line = round(predicted_total / 5) * 5  # round to nearest 5
        # Over/Under: simple probability based on predicted vs line
        over_prob = 0.55 if predicted_total > runs_line else 0.45

        # --- First innings prediction ---
        first_innings = bench * (1 + (team1_form.form_score - 0.5) * 0.2)
        fi_low = int(first_innings * 0.88)
        fi_high = int(first_innings * 1.12)

        # --- Recommendation ---
        if max(t1_win, t2_win) >= 0.62:
            winner = team1 if t1_win > t2_win else team2
            bet = f"{winner} to Win"
            market = "Match Winner"
            best_prob = max(t1_win, t2_win)
        else:
            bet = f"Over {runs_line} Runs"
            market = "Total Runs"
            best_prob = over_prob

        # --- Confidence ---
        prob_gap = abs(t1_win - t2_win)
        form_factor = min(1.0, (team1_form.matches_played + team2_form.matches_played) / 10)
        h2h_factor = min(1.0, h2h.matches_played / 8)
        confidence = (0.50 * prob_gap / 0.5) + (0.30 * form_factor) + (0.20 * h2h_factor)
        confidence = max(0.0, min(1.0, confidence))

        is_value, ev = _value_bet(best_prob)

        # --- Player predictions (premium) ---
        all_batters = [p for p in (team1_squad or []) + (team2_squad or [])
                       if p.role in ("batsman", "allrounder", "wicketkeeper")]
        all_bowlers = [p for p in (team1_squad or []) + (team2_squad or [])
                       if p.role in ("bowler", "allrounder")]

        return CricketPrediction(
            match_id=match_id,
            team1=team1,
            team2=team2,
            team1_logo=team1_logo,
            team2_logo=team2_logo,
            match_start_utc=match_start_utc,
            series_name=series_name,
            format=CricketFormat.T20,
            team1_win_prob=round(t1_win, 4),
            team2_win_prob=round(t2_win, 4),
            draw_prob=0.0,
            team1_win_prob_post_toss=round(t1_win_post, 4) if toss_winner else None,
            team2_win_prob_post_toss=round(t2_win_post, 4) if toss_winner else None,
            total_runs_line=float(runs_line),
            over_line_prob=round(over_prob, 4),
            under_line_prob=round(1.0 - over_prob, 4),
            predicted_first_innings_score=round(first_innings, 1),
            predicted_first_innings_range=(fi_low, fi_high),
            top_batsman_predictions=_top_player_predictions(all_batters, "bat"),
            top_bowler_predictions=_top_player_predictions(all_bowlers, "bowl"),
            recommended_bet=bet,
            recommended_market=market,
            confidence=round(confidence, 4),
            confidence_label=_confidence_label(confidence),
            is_value_bet=is_value,
            value_bet_market=bet if is_value else None,
            expected_value=ev if is_value else None,
            team1_form=team1_form.last_5,
            team2_form=team2_form.last_5,
            h2h_team1_wins=h2h.team1_wins,
            h2h_team2_wins=h2h.team2_wins,
            h2h_draws=h2h.no_results,
            venue_name=venue.venue_name if venue else None,
            pitch_type=venue.pitch_type if venue else None,
            dew_factor=dew,
        )


# ---------------------------------------------------------------------------
# ODI prediction engine
# ---------------------------------------------------------------------------

class ODIPredictionEngine:
    """
    ODI model: similar to T20 but with powerplay weighting and higher run line.
    Draws don't exist in limited-overs cricket (DLS/Super Over applies).
    """
    VENUE_BENCHMARK = 265.0

    def predict(
        self,
        match_id: str,
        team1: str,
        team2: str,
        team1_logo: str | None,
        team2_logo: str | None,
        match_start_utc: datetime,
        series_name: str,
        team1_form: TeamBattingForm,
        team2_form: TeamBattingForm,
        h2h: HeadToHead,
        venue: VenueStats | None = None,
        toss_winner: str | None = None,
        toss_decision: str | None = None,
        team1_squad: list[PlayerForm] | None = None,
        team2_squad: list[PlayerForm] | None = None,
    ) -> CricketPrediction:
        bench = (venue.avg_first_innings_score_odi if venue else self.VENUE_BENCHMARK)
        dew = venue.dew_factor if venue else False

        if team1_form.avg_runs_scored:
            t1_avg_scored = team1_form.avg_runs_scored
            t1_avg_conceded = team1_form.avg_runs_conceded or bench
        else:
            t1_avg_scored, t1_avg_conceded = _synthetic_run_rates(team1, bench)

        if team2_form.avg_runs_scored:
            t2_avg_scored = team2_form.avg_runs_scored
            t2_avg_conceded = team2_form.avg_runs_conceded or bench
        else:
            t2_avg_scored, t2_avg_conceded = _synthetic_run_rates(team2, bench)

        t1_win, t2_win = _win_prob_from_run_rates(
            t1_avg_scored, t1_avg_conceded, t2_avg_scored, t2_avg_conceded, bench
        )

        form_diff = team1_form.form_score - team2_form.form_score
        t1_win = max(0.05, min(0.95, t1_win + form_diff * 0.10))
        t2_win = 1.0 - t1_win
        t1_win, t2_win = _blend_h2h(t1_win, t2_win, h2h)
        t1_win_post, t2_win_post = _toss_adjustment(t1_win, t2_win, toss_winner or "", toss_decision or "", team1, dew)

        avg_team_score = (t1_avg_scored + t2_avg_scored) / 2
        predicted_total = avg_team_score + (avg_team_score * 0.95)
        runs_line = round(predicted_total / 10) * 10
        over_prob = 0.55 if predicted_total > runs_line else 0.45

        first_innings = bench * (1 + (team1_form.form_score - 0.5) * 0.15)
        fi_low = int(first_innings * 0.90)
        fi_high = int(first_innings * 1.10)

        if max(t1_win, t2_win) >= 0.60:
            winner = team1 if t1_win > t2_win else team2
            bet = f"{winner} to Win"
            market = "Match Winner"
            best_prob = max(t1_win, t2_win)
        else:
            bet = f"Over {runs_line} Runs"
            market = "Total Runs"
            best_prob = over_prob

        prob_gap = abs(t1_win - t2_win)
        form_factor = min(1.0, (team1_form.matches_played + team2_form.matches_played) / 10)
        h2h_factor = min(1.0, h2h.matches_played / 8)
        confidence = (0.50 * prob_gap / 0.5) + (0.30 * form_factor) + (0.20 * h2h_factor)
        confidence = max(0.0, min(1.0, confidence))

        is_value, ev = _value_bet(best_prob)

        all_batters = [p for p in (team1_squad or []) + (team2_squad or [])
                       if p.role in ("batsman", "allrounder", "wicketkeeper")]
        all_bowlers = [p for p in (team1_squad or []) + (team2_squad or [])
                       if p.role in ("bowler", "allrounder")]

        return CricketPrediction(
            match_id=match_id,
            team1=team1,
            team2=team2,
            team1_logo=team1_logo,
            team2_logo=team2_logo,
            match_start_utc=match_start_utc,
            series_name=series_name,
            format=CricketFormat.ODI,
            team1_win_prob=round(t1_win, 4),
            team2_win_prob=round(t2_win, 4),
            draw_prob=0.0,
            team1_win_prob_post_toss=round(t1_win_post, 4) if toss_winner else None,
            team2_win_prob_post_toss=round(t2_win_post, 4) if toss_winner else None,
            total_runs_line=float(runs_line),
            over_line_prob=round(over_prob, 4),
            under_line_prob=round(1.0 - over_prob, 4),
            predicted_first_innings_score=round(first_innings, 1),
            predicted_first_innings_range=(fi_low, fi_high),
            top_batsman_predictions=_top_player_predictions(all_batters, "bat"),
            top_bowler_predictions=_top_player_predictions(all_bowlers, "bowl"),
            recommended_bet=bet,
            recommended_market=market,
            confidence=round(confidence, 4),
            confidence_label=_confidence_label(confidence),
            is_value_bet=is_value,
            value_bet_market=bet if is_value else None,
            expected_value=ev if is_value else None,
            team1_form=team1_form.last_5,
            team2_form=team2_form.last_5,
            h2h_team1_wins=h2h.team1_wins,
            h2h_team2_wins=h2h.team2_wins,
            h2h_draws=h2h.no_results,
            venue_name=venue.venue_name if venue else None,
            pitch_type=venue.pitch_type if venue else None,
            dew_factor=dew,
        )


# ---------------------------------------------------------------------------
# Test prediction engine
# ---------------------------------------------------------------------------

class TestPredictionEngine:
    """
    Test match model: draw is a real outcome (typically 25-35% of Tests end in draws).
    Win probabilities derived from batting averages + bowling strike rate.
    No toss dew factor for Tests (multi-day pitches change conditions differently).
    """
    VENUE_BENCHMARK = 320.0
    BASE_DRAW_PROB = 0.28  # historical draw rate in Test cricket

    def predict(
        self,
        match_id: str,
        team1: str,
        team2: str,
        team1_logo: str | None,
        team2_logo: str | None,
        match_start_utc: datetime,
        series_name: str,
        team1_form: TeamBattingForm,
        team2_form: TeamBattingForm,
        h2h: HeadToHead,
        venue: VenueStats | None = None,
        team1_squad: list[PlayerForm] | None = None,
        team2_squad: list[PlayerForm] | None = None,
    ) -> CricketPrediction:
        bench = (venue.avg_first_innings_score_test if venue else self.VENUE_BENCHMARK)

        if team1_form.avg_runs_scored:
            t1_avg_scored = team1_form.avg_runs_scored
            t1_avg_conceded = team1_form.avg_runs_conceded or bench
        else:
            t1_avg_scored, t1_avg_conceded = _synthetic_run_rates(team1, bench)

        if team2_form.avg_runs_scored:
            t2_avg_scored = team2_form.avg_runs_scored
            t2_avg_conceded = team2_form.avg_runs_conceded or bench
        else:
            t2_avg_scored, t2_avg_conceded = _synthetic_run_rates(team2, bench)

        raw_t1_win, raw_t2_win = _win_prob_from_run_rates(
            t1_avg_scored, t1_avg_conceded, t2_avg_scored, t2_avg_conceded, bench
        )

        # Pitch type affects draw probability: spin-friendly pitches → more draws
        draw_prob = self.BASE_DRAW_PROB
        if venue and venue.pitch_type == "spin-friendly":
            draw_prob = 0.35
        elif venue and venue.pitch_type == "pace-friendly":
            draw_prob = 0.22

        # Adjust win probs to account for draw probability
        decisive_prob = 1.0 - draw_prob
        form_diff = team1_form.form_score - team2_form.form_score
        raw_t1_win = max(0.05, min(0.90, raw_t1_win + form_diff * 0.08))
        raw_t2_win = 1.0 - raw_t1_win

        t1_win = raw_t1_win * decisive_prob
        t2_win = raw_t2_win * decisive_prob

        t1_win, t2_win = _blend_h2h(t1_win, t2_win, h2h)

        # Re-normalise to ensure t1+t2+draw = 1
        total = t1_win + t2_win + draw_prob
        t1_win /= total
        t2_win /= total
        draw_prob /= total

        first_innings = bench * (1 + (team1_form.form_score - 0.5) * 0.12)
        fi_low = int(first_innings * 0.85)
        fi_high = int(first_innings * 1.15)

        # Total runs O/U (both teams' first innings combined)
        runs_line = round((bench * 2) / 50) * 50
        over_prob = 0.50  # Test totals are harder to predict

        max_decisive = max(t1_win, t2_win)
        if max_decisive >= 0.55:
            winner = team1 if t1_win > t2_win else team2
            bet = f"{winner} to Win"
            market = "Match Winner"
            best_prob = max_decisive
        elif draw_prob >= 0.35:
            bet = "Draw"
            market = "Match Winner"
            best_prob = draw_prob
        else:
            bet = f"Over {runs_line} Runs (Combined)"
            market = "Total Runs"
            best_prob = over_prob

        prob_gap = max(t1_win, t2_win, draw_prob) - sorted([t1_win, t2_win, draw_prob])[-2]
        form_factor = min(1.0, (team1_form.matches_played + team2_form.matches_played) / 6)
        h2h_factor = min(1.0, h2h.matches_played / 5)
        confidence = (0.50 * prob_gap / 0.4) + (0.30 * form_factor) + (0.20 * h2h_factor)
        confidence = max(0.0, min(1.0, confidence))

        is_value, ev = _value_bet(best_prob)

        all_batters = [p for p in (team1_squad or []) + (team2_squad or [])
                       if p.role in ("batsman", "allrounder", "wicketkeeper")]
        all_bowlers = [p for p in (team1_squad or []) + (team2_squad or [])
                       if p.role in ("bowler", "allrounder")]

        return CricketPrediction(
            match_id=match_id,
            team1=team1,
            team2=team2,
            team1_logo=team1_logo,
            team2_logo=team2_logo,
            match_start_utc=match_start_utc,
            series_name=series_name,
            format=CricketFormat.TEST,
            team1_win_prob=round(t1_win, 4),
            team2_win_prob=round(t2_win, 4),
            draw_prob=round(draw_prob, 4),
            total_runs_line=float(runs_line),
            over_line_prob=round(over_prob, 4),
            under_line_prob=round(1.0 - over_prob, 4),
            predicted_first_innings_score=round(first_innings, 1),
            predicted_first_innings_range=(fi_low, fi_high),
            top_batsman_predictions=_top_player_predictions(all_batters, "bat"),
            top_bowler_predictions=_top_player_predictions(all_bowlers, "bowl"),
            recommended_bet=bet,
            recommended_market=market,
            confidence=round(confidence, 4),
            confidence_label=_confidence_label(confidence),
            is_value_bet=is_value,
            value_bet_market=bet if is_value else None,
            expected_value=ev if is_value else None,
            team1_form=team1_form.last_5,
            team2_form=team2_form.last_5,
            h2h_team1_wins=h2h.team1_wins,
            h2h_team2_wins=h2h.team2_wins,
            h2h_draws=h2h.no_results,
            venue_name=venue.venue_name if venue else None,
            pitch_type=venue.pitch_type if venue else None,
            dew_factor=False,
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_engine(fmt: CricketFormat):
    if fmt == CricketFormat.T20 or fmt == CricketFormat.T10:
        return T20PredictionEngine()
    if fmt == CricketFormat.ODI:
        return ODIPredictionEngine()
    return TestPredictionEngine()
