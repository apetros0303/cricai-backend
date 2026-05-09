"""
Parses raw CricAPI responses into our domain models.
Acts as the translation layer between the external API and our prediction engine.
"""

import logging
from datetime import datetime, timezone, timedelta
from models.match import (
    CricketMatch, CricketFormat, MatchStatus, TeamInfo, VenueStats,
    TeamBattingForm, HeadToHead, PlayerForm
)
from services.cricapi_client import CricApiClient

logger = logging.getLogger(__name__)

# Maps the fake placeholder UUIDs in settings to series name search patterns.
# When CricAPI series_info fails, we fall back to searching upcoming matches by name.
_SERIES_NAME_PATTERNS: dict[str, list[str]] = {
    "d5a498c8-7596-4b93-8ab0-e0efc3345312": ["ipl", "indian premier league", "indian premier"],
    "3e1e7a33-b944-44c4-9beb-d0da55fa95bc": ["psl", "pakistan super league"],
    "a5c5e6f8-1b34-4c78-8fe2-d1d34a5f6612": ["big bash", "bbl"],
    "d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a": ["t20 world cup", "icc men's t20"],
    "a1b2c3d4-e5f6-7890-abcd-ef1234567890": ["odi world cup", "icc men's cricket world cup"],
    "b2c3d4e5-f6a7-8901-bcde-f12345678901": ["world test championship", "wtc"],
}

_FORMAT_MAP = {
    "t20": CricketFormat.T20,
    "t20i": CricketFormat.T20,
    "ipl": CricketFormat.T20,
    "psl": CricketFormat.T20,
    "bbl": CricketFormat.T20,
    "t10": CricketFormat.T10,
    "odi": CricketFormat.ODI,
    "odm": CricketFormat.ODI,
    "test": CricketFormat.TEST,
}

_STATUS_MAP = {
    # Finished — check before LIVE so "won by" beats generic substring matches
    "won by": MatchStatus.FINISHED,
    "wins by": MatchStatus.FINISHED,
    "win by": MatchStatus.FINISHED,
    "no result": MatchStatus.FINISHED,
    "complete": MatchStatus.FINISHED,
    "finished": MatchStatus.FINISHED,
    "abandoned": MatchStatus.CANCELLED,
    "cancelled": MatchStatus.CANCELLED,
    # Live — broad coverage of CricAPI status strings
    "in progress": MatchStatus.LIVE,
    "live": MatchStatus.LIVE,
    "innings": MatchStatus.LIVE,   # covers "1st innings", "2nd innings", "4th innings"
    "batting": MatchStatus.LIVE,
    "bowling": MatchStatus.LIVE,
    "opt to bat": MatchStatus.LIVE,
    "elected to bat": MatchStatus.LIVE,
    "elected to field": MatchStatus.LIVE,
    " ov)": MatchStatus.LIVE,      # score lines like "145/4 (15.2 ov)"
    " ov ": MatchStatus.LIVE,
    "rain": MatchStatus.RAIN_DELAY,
    # Scheduled
    "match not started": MatchStatus.SCHEDULED,
    "not started": MatchStatus.SCHEDULED,
    "match starts": MatchStatus.SCHEDULED,
    "toss": MatchStatus.SCHEDULED,
}


def _parse_format(raw: str, name: str = "") -> CricketFormat:
    key = raw.lower().replace(" ", "")
    for k, v in _FORMAT_MAP.items():
        if k in key:
            return v
    # Fallback: infer from match name
    name_lower = name.lower()
    if "test" in name_lower:
        return CricketFormat.TEST
    if " odi" in name_lower or "one day" in name_lower:
        return CricketFormat.ODI
    if "t10" in name_lower:
        return CricketFormat.T10
    return CricketFormat.T20


def _parse_status(raw: str) -> MatchStatus:
    key = raw.lower()
    for k, v in _STATUS_MAP.items():
        if k in key:
            return v
    return MatchStatus.SCHEDULED


def _parse_datetime(raw: str) -> datetime:
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


class CricketService:
    def __init__(self):
        self.client = CricApiClient()

    def _build_match(self, raw: dict) -> CricketMatch | None:
        try:
            teams = raw.get("teams", [])
            team_info = raw.get("teamInfo", [])

            def _team(name: str) -> TeamInfo:
                for t in team_info:
                    if t.get("name", "").lower() == name.lower():
                        return TeamInfo(
                            id=t.get("shortname", name),
                            name=t.get("name", name),
                            short_name=t.get("shortname"),
                            logo_url=t.get("img"),
                        )
                return TeamInfo(id=name, name=name)

            team1 = _team(teams[0]) if len(teams) > 0 else TeamInfo(id="TBA", name="TBA")
            team2 = _team(teams[1]) if len(teams) > 1 else TeamInfo(id="TBA", name="TBA")

            venue_raw = raw.get("venue", "")
            venue = VenueStats(venue_name=venue_raw) if venue_raw else None

            score_list = raw.get("score", [])
            live1 = score_list[0].get("r", "") if len(score_list) > 0 else None
            live2 = score_list[1].get("r", "") if len(score_list) > 1 else None
            # Current over comes from the most recent innings entry
            current_over: float | None = None
            for sc in reversed(score_list):
                o = sc.get("o")
                if o is not None:
                    try:
                        current_over = float(o)
                        break
                    except (ValueError, TypeError):
                        pass

            return CricketMatch(
                id=raw.get("id", ""),
                name=raw.get("name", ""),
                team1=team1,
                team2=team2,
                match_start_utc=_parse_datetime(raw.get("dateTimeGMT", "")),
                series_id=raw.get("series_id"),
                series_name=raw.get("series", raw.get("name", "")),
                format=_parse_format(raw.get("matchType", ""), raw.get("name", "")),
                venue=venue,
                status=_parse_status(raw.get("status", "")),
                live_score_team1=str(live1) if live1 else None,
                live_score_team2=str(live2) if live2 else None,
                live_current_over=current_over,
                toss_winner=raw.get("tossWinner"),
                toss_decision=raw.get("tossChoice"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse match {raw.get('id')}: {e}")
            return None

    async def get_upcoming_matches(self, series_id: str | None = None) -> list[CricketMatch]:
        """Fetch upcoming + in-progress matches from both endpoints, deduped."""
        # currentMatches = live/in-progress; matches = scheduled upcoming
        current_raw: list[dict] = []
        scheduled_raw: list[dict] = []
        try:
            current_raw = await self.client.get_current_matches()
        except Exception as e:
            logger.warning(f"currentMatches fetch failed: {e}")
        try:
            scheduled_raw = await self.client.get_matches()
        except Exception as e:
            logger.warning(f"matches fetch failed: {e}")

        now_utc = datetime.utcnow()
        cutoff = now_utc - timedelta(hours=3)

        seen: set[str] = set()
        matches: list[CricketMatch] = []

        # Process current matches — never drop by stale-scheduled rule (they're live)
        for raw in current_raw:
            mid = raw.get("id", "")
            if mid in seen:
                continue
            seen.add(mid)
            m = self._build_match(raw)
            if not m:
                continue
            if m.status in (MatchStatus.FINISHED, MatchStatus.CANCELLED):
                continue
            if series_id and m.series_id != series_id:
                continue
            matches.append(m)

        # Process upcoming scheduled matches
        for raw in scheduled_raw:
            mid = raw.get("id", "")
            if mid in seen:
                continue
            seen.add(mid)
            m = self._build_match(raw)
            if not m:
                continue
            if m.status in (MatchStatus.FINISHED, MatchStatus.CANCELLED):
                continue
            # Drop stale scheduled matches (started >3h ago but still showing as scheduled)
            if m.status == MatchStatus.SCHEDULED and m.match_start_utc < cutoff:
                continue
            if series_id and m.series_id != series_id:
                continue
            matches.append(m)

        return matches

    async def get_live_matches(self) -> list[CricketMatch]:
        try:
            raw_list = await self.client.get_current_matches()
        except Exception as e:
            logger.error(f"Failed to fetch live matches: {e}")
            return []
        matches = []
        for raw in raw_list:
            m = self._build_match(raw)
            if m and m.is_live:
                matches.append(m)
        return matches

    async def get_match_by_id(self, match_id: str) -> CricketMatch | None:
        try:
            raw = await self.client.get_match_info(match_id)
            if not raw:
                return None
            return self._build_match(raw)
        except Exception as e:
            logger.error(f"Failed to fetch match {match_id}: {e}")
            return None

    async def get_series_matches(self, series_id: str) -> list[CricketMatch]:
        # Try CricAPI series_info first
        raw_list: list[dict] = []
        try:
            raw_list = await self.client.get_series_matches(series_id)
        except Exception as e:
            logger.warning(f"CricAPI series_info failed for {series_id}: {e}")

        matches: list[CricketMatch] = []
        for raw in raw_list:
            m = self._build_match(raw)
            if m:
                matches.append(m)

        if matches:
            return matches

        # Fallback 1: search upcoming feed by series_id field
        logger.info(f"Falling back to upcoming-feed search for series {series_id}")
        upcoming = await self.get_upcoming_matches()
        by_id = [m for m in upcoming if m.series_id == series_id]
        if by_id:
            return by_id

        # Fallback 2: name-pattern search (handles placeholder UUIDs in settings)
        patterns = _SERIES_NAME_PATTERNS.get(series_id, [])
        if patterns:
            name_matched = [
                m for m in upcoming
                if any(p in m.series_name.lower() or p in m.name.lower() for p in patterns)
            ]
            if name_matched:
                logger.info(f"Found {len(name_matched)} matches via name pattern for {series_id}")
                return name_matched

        return []

    def _extract_innings_runs(self, scorecard: dict, team_name: str) -> tuple[int | None, int | None]:
        """
        Returns (runs_scored_by_team, runs_scored_against_team) from a scorecard.
        CricAPI scorecard structure: data.scorecard = list of innings dicts with keys:
          inning (str, e.g. "Mumbai Indians Inning 1"), r (runs), w (wickets), o (overs)
        """
        innings = scorecard.get("scorecard", [])
        team_key = team_name.lower()
        batting_runs: list[int] = []
        bowling_runs: list[int] = []

        for inn in innings:
            inning_label = inn.get("inning", "").lower()
            runs = inn.get("r")
            if runs is None:
                continue
            try:
                runs = int(runs)
            except (ValueError, TypeError):
                continue
            if team_key in inning_label:
                batting_runs.append(runs)
            else:
                bowling_runs.append(runs)

        scored = batting_runs[0] if batting_runs else None
        conceded = bowling_runs[0] if bowling_runs else None
        return scored, conceded

    def _determine_winner(self, scorecard: dict, match_name: str) -> str | None:
        """
        Extract winner from scorecard. CricAPI puts winner in:
          data.winner (direct field) OR derives from 'matchWinner' OR status string.
        Returns winner team name string, or None if no result / abandoned.
        """
        # Direct winner field (available on paid plan)
        winner = scorecard.get("winner") or scorecard.get("matchWinner")
        if winner:
            return str(winner).strip()
        # Fall back to status string: "Mumbai Indians won by 5 runs"
        status = scorecard.get("status", "")
        if "won by" in status.lower():
            return status.split(" won by")[0].strip()
        if "no result" in status.lower() or "abandoned" in status.lower():
            return None
        return None

    async def get_team_form(
        self,
        team_name: str,
        recent_matches: list[CricketMatch],
        last_n: int = 10,
    ) -> TeamBattingForm:
        """
        Build real TeamBattingForm by fetching scorecards for each finished match
        involving this team. Paid plan allows scorecard access; results are
        cached 7 days so repeat calls for same matches are free.
        """
        team_key = team_name.lower()
        wins = losses = no_results = 0
        scores_batting: list[float] = []
        scores_bowling: list[float] = []
        last_5: list[str] = []
        processed = 0

        # Filter and sort: most recent finished matches first
        relevant = [
            m for m in recent_matches
            if m.status == MatchStatus.FINISHED
            and team_key in m.name.lower()
        ]

        for m in relevant[:last_n]:
            try:
                sc = await self.client.get_match_scorecard(m.id)
                if not sc:
                    continue

                scored, conceded = self._extract_innings_runs(sc, team_name)
                winner = self._determine_winner(sc, m.name)

                if scored is not None:
                    scores_batting.append(float(scored))
                if conceded is not None:
                    scores_bowling.append(float(conceded))

                if winner is None:
                    no_results += 1
                    last_5.append("N")
                elif team_key in winner.lower():
                    wins += 1
                    last_5.append("W")
                else:
                    losses += 1
                    last_5.append("L")

                processed += 1
            except Exception as e:
                logger.warning(f"Scorecard parse failed for {m.id}: {e}")
                continue

        logger.info(
            f"Team form for {team_name}: {processed} matches processed, "
            f"W{wins}/L{losses}/N{no_results}"
        )

        return TeamBattingForm(
            team_id=team_name,
            team_name=team_name,
            matches_played=processed,
            wins=wins,
            losses=losses,
            avg_runs_scored=round(sum(scores_batting) / len(scores_batting), 1) if scores_batting else 0.0,
            avg_runs_conceded=round(sum(scores_bowling) / len(scores_bowling), 1) if scores_bowling else 0.0,
            last_5=last_5[-5:],
        )

    async def get_head_to_head(
        self,
        team1_name: str,
        team2_name: str,
        recent_matches: list[CricketMatch],
        last_n: int = 10,
    ) -> HeadToHead:
        """
        Build real H2H by fetching scorecards for finished matches between
        the two specific teams. Uses 7-day scorecard cache.
        """
        t1_key = team1_name.lower()
        t2_key = team2_name.lower()

        h2h_matches = [
            m for m in recent_matches
            if m.status == MatchStatus.FINISHED
            and t1_key in m.name.lower()
            and t2_key in m.name.lower()
        ]

        t1_wins = t2_wins = no_results = 0
        first_innings_scores: list[float] = []
        recent_results: list[str] = []

        for m in h2h_matches[:last_n]:
            try:
                sc = await self.client.get_match_scorecard(m.id)
                if not sc:
                    continue

                # First innings score for run-line context
                innings = sc.get("scorecard", [])
                if innings:
                    fi_runs = innings[0].get("r")
                    if fi_runs is not None:
                        try:
                            first_innings_scores.append(float(fi_runs))
                        except (ValueError, TypeError):
                            pass

                winner = self._determine_winner(sc, m.name)
                if winner is None:
                    no_results += 1
                    recent_results.append("N")
                elif t1_key in winner.lower():
                    t1_wins += 1
                    recent_results.append("T1")
                elif t2_key in winner.lower():
                    t2_wins += 1
                    recent_results.append("T2")
                else:
                    no_results += 1
                    recent_results.append("N")
            except Exception as e:
                logger.warning(f"H2H scorecard parse failed for {m.id}: {e}")
                continue

        return HeadToHead(
            matches_played=t1_wins + t2_wins + no_results,
            team1_wins=t1_wins,
            team2_wins=t2_wins,
            no_results=no_results,
            avg_first_innings_score=round(
                sum(first_innings_scores) / len(first_innings_scores), 1
            ) if first_innings_scores else 0.0,
            recent_results=recent_results[-5:],
        )

    async def get_squad_form(self, match_id: str) -> tuple[list[PlayerForm], list[PlayerForm]]:
        """
        Fetch squad lists from the match scorecard and build PlayerForm
        objects using career stats from players_info endpoint.
        Returns (team1_players, team2_players).
        """
        sc = await self.client.get_match_scorecard(match_id)
        if not sc:
            return [], []

        teams_data = sc.get("teams", [])
        if len(teams_data) < 2:
            return [], []

        async def _build_player_form(player_raw: dict) -> PlayerForm | None:
            pid = player_raw.get("id") or player_raw.get("player_id")
            name = player_raw.get("name", "")
            if not pid or not name:
                return None
            try:
                info = await self.client.get_player_info(pid)
                if not info:
                    return PlayerForm(player_id=pid, player_name=name, role="batsman")

                role = info.get("role", "batsman").lower()
                stats = info.get("stats", [])

                batting_avg = bowling_avg = bowling_economy = 0.0
                batting_sr = bowling_sr = 0.0

                for s in stats:
                    fn = s.get("fn", "").lower()
                    if "batting" in fn:
                        batting_avg = float(s.get("avg") or 0)
                        batting_sr = float(s.get("sr") or 0)
                    elif "bowling" in fn:
                        bowling_avg = float(s.get("avg") or 0)
                        bowling_economy = float(s.get("econ") or 0)
                        bowling_sr = float(s.get("sr") or 0)

                return PlayerForm(
                    player_id=pid,
                    player_name=name,
                    role=role if role in ("batsman", "bowler", "allrounder", "wicketkeeper") else "batsman",
                    batting_avg=batting_avg,
                    batting_strike_rate=batting_sr,
                    bowling_avg=bowling_avg,
                    bowling_economy=bowling_economy,
                    bowling_strike_rate=bowling_sr,
                )
            except Exception as e:
                logger.warning(f"Player info fetch failed for {pid}: {e}")
                return PlayerForm(player_id=str(pid), player_name=name, role="batsman")

        team1_players: list[PlayerForm] = []
        team2_players: list[PlayerForm] = []

        for raw_player in teams_data[0].get("players", []):
            pf = await _build_player_form(raw_player)
            if pf:
                team1_players.append(pf)

        for raw_player in teams_data[1].get("players", []):
            pf = await _build_player_form(raw_player)
            if pf:
                team2_players.append(pf)

        return team1_players, team2_players
