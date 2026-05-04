from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from models.match import CricketFormat


class PlayerPrediction(BaseModel):
    """Prediction for a specific player market (top bat/bowl)."""
    player_id: str
    player_name: str
    probability: float = Field(ge=0.0, le=1.0)
    predicted_runs: Optional[float] = None
    predicted_wickets: Optional[float] = None
    form_score: float = 0.5
    confidence_label: str = "Medium"


class CricketPrediction(BaseModel):
    match_id: str
    team1: str
    team2: str
    team1_logo: Optional[str] = None
    team2_logo: Optional[str] = None
    match_start_utc: datetime
    series_name: str
    format: CricketFormat

    # Match winner
    team1_win_prob: float
    team2_win_prob: float
    draw_prob: float = 0.0          # Only relevant for Test matches

    # Toss-adjusted probabilities (recalculated when toss is known)
    team1_win_prob_post_toss: Optional[float] = None
    team2_win_prob_post_toss: Optional[float] = None

    # Total runs O/U (format-specific line)
    total_runs_line: float          # e.g. 165.5 for T20, 280.5 for ODI
    over_line_prob: float = 0.0
    under_line_prob: float = 0.0

    # First innings score range
    predicted_first_innings_score: float = 0.0
    predicted_first_innings_range: tuple[int, int] = (0, 0)  # (low, high)

    # Player predictions (PREMIUM)
    top_batsman_predictions: list[PlayerPrediction] = Field(default_factory=list)
    top_bowler_predictions: list[PlayerPrediction] = Field(default_factory=list)

    # Recommendation
    recommended_bet: str
    recommended_market: str         # "Match Winner", "Total Runs", "Top Batsman"
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: str           # "Low", "Medium", "High"

    # Value bet (PREMIUM)
    is_value_bet: bool = False
    value_bet_market: Optional[str] = None
    expected_value: Optional[float] = None

    # Team form strings for display
    team1_form: list[str] = Field(default_factory=list)   # ["W","L","W","W","L"]
    team2_form: list[str] = Field(default_factory=list)

    # H2H summary
    h2h_team1_wins: int = 0
    h2h_team2_wins: int = 0
    h2h_draws: int = 0

    # Venue context
    venue_name: Optional[str] = None
    pitch_type: Optional[str] = None
    dew_factor: bool = False

    # AI Analysis (PREMIUM, multi-language)
    ai_analysis: Optional[str] = None
    ai_key_factors: list[str] = Field(default_factory=list)
    ai_confidence: Optional[str] = None
    language: str = "en"            # "en", "hi", "ur"

    model: str = "cricket_v1"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
