from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional


class CricketFormat(str, Enum):
    T20 = "t20"
    ODI = "odi"
    TEST = "test"
    T10 = "t10"


class MatchStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    RAIN_DELAY = "rain_delay"


class TeamInfo(BaseModel):
    id: str
    name: str
    short_name: Optional[str] = None
    logo_url: Optional[str] = None
    country: Optional[str] = None


class VenueStats(BaseModel):
    """Historical averages at a specific ground — used to adjust run predictions."""
    venue_id: Optional[str] = None
    venue_name: str
    city: Optional[str] = None
    country: Optional[str] = None
    avg_first_innings_score_t20: float = 160.0
    avg_first_innings_score_odi: float = 265.0
    avg_first_innings_score_test: float = 320.0
    # Pitch type affects spin/pace bowling weighting
    pitch_type: Optional[str] = None   # "spin-friendly", "pace-friendly", "balanced"
    dew_factor: bool = False            # evening dew often helps chasing team


class PlayerForm(BaseModel):
    """Recent batting/bowling performance used in player predictions."""
    player_id: str
    player_name: str
    role: str   # "batsman", "bowler", "allrounder", "wicketkeeper"
    matches_played: int = 0

    # Batting
    batting_avg: float = 0.0
    batting_strike_rate: float = 0.0
    recent_scores: list[int] = Field(default_factory=list)  # last 5 innings

    # Bowling
    bowling_avg: float = 0.0
    bowling_economy: float = 0.0
    bowling_strike_rate: float = 0.0
    recent_wickets: list[int] = Field(default_factory=list)  # last 5 games

    @property
    def batting_form_score(self) -> float:
        """0-1 based on last 5 scores relative to personal average."""
        if not self.recent_scores or self.batting_avg == 0:
            return 0.5
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        score = 0.0
        for i, runs in enumerate(self.recent_scores[:5]):
            w = weights[i] if i < len(weights) else 0.05
            ratio = min(runs / max(self.batting_avg, 1), 2.0) / 2.0
            score += w * ratio
        return round(score, 3)

    @property
    def bowling_form_score(self) -> float:
        """0-1 based on recent wickets (more = better)."""
        if not self.recent_wickets:
            return 0.5
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        score = 0.0
        for i, wkts in enumerate(self.recent_wickets[:5]):
            w = weights[i] if i < len(weights) else 0.05
            # 3 wickets = excellent, scale to 1.0
            score += w * min(wkts / 3.0, 1.0)
        return round(score, 3)


class TeamBattingForm(BaseModel):
    """Aggregate batting stats for a team over recent matches."""
    team_id: str
    team_name: str
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    avg_runs_scored: float = 0.0
    avg_runs_conceded: float = 0.0
    avg_wickets_lost: float = 0.0
    avg_wickets_taken: float = 0.0
    last_5: list[str] = Field(default_factory=list)   # "W" / "L" / "N" (no result)

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.5
        return self.wins / self.matches_played

    @property
    def form_score(self) -> float:
        if not self.last_5:
            return 0.5
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        score = 0.0
        for i, result in enumerate(self.last_5[:5]):
            w = weights[i] if i < len(weights) else 0.05
            if result == "W":
                score += w
        return round(score, 3)


class HeadToHead(BaseModel):
    matches_played: int = 0
    team1_wins: int = 0
    team2_wins: int = 0
    no_results: int = 0
    avg_first_innings_score: float = 0.0
    recent_results: list[str] = Field(default_factory=list)   # "T1", "T2", "N"


class CricketMatch(BaseModel):
    id: str
    name: str
    team1: TeamInfo
    team2: TeamInfo
    match_start_utc: datetime
    series_id: Optional[str] = None
    series_name: str
    format: CricketFormat
    venue: Optional[VenueStats] = None
    status: MatchStatus = MatchStatus.SCHEDULED

    # Live score fields (populated when status=live)
    live_score_team1: Optional[str] = None   # e.g. "142/4 (15.2)"
    live_score_team2: Optional[str] = None
    live_current_over: Optional[float] = None
    live_batting_team: Optional[str] = None
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None      # "bat" / "field"
