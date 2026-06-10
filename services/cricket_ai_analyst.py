"""
Cricket AI analyst — PREMIUM feature, multi-language.
Generates match analysis in English (en), Hindi (hi), or Urdu (ur).
"""

import anthropic
import logging
import json
import re
from cachetools import TTLCache
from models.match import TeamBattingForm, HeadToHead, CricketFormat
from models.prediction import CricketPrediction
from config.settings import get_settings

# Cache AI analysis per (match_id, language) for 6 hours — analysis is stable within a day
_ai_cache: TTLCache = TTLCache(maxsize=60, ttl=21600)

logger = logging.getLogger(__name__)

_LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in English.",
    "hi": (
        "Respond entirely in Hindi (Devanagari script). "
        "Use clear, simple Hindi suitable for a general sports audience. "
        "All JSON field values must be in Hindi."
    ),
    "ur": (
        "Respond entirely in Urdu (Nastaliq/Arabic script). "
        "Use clear, simple Urdu suitable for a Pakistani sports audience. "
        "All JSON field values must be in Urdu."
    ),
}

_FORMAT_LABELS = {
    CricketFormat.T20: "T20",
    CricketFormat.ODI: "ODI",
    CricketFormat.TEST: "Test Match",
    CricketFormat.T10: "T10",
}


def _extract_json(raw: str) -> dict:
    text = raw
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise json.JSONDecodeError("No JSON object found", raw, 0)


class CricketAiAnalyst:
    def __init__(self):
        settings = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-haiku-4-5-20251001"

    async def analyze_match(
        self,
        team1: str,
        team2: str,
        series: str,
        fmt: CricketFormat,
        team1_form: TeamBattingForm,
        team2_form: TeamBattingForm,
        h2h: HeadToHead,
        prediction: CricketPrediction,
        language: str = "en",
        venue_notes: str = "",
    ) -> tuple[str, list[str], str]:
        """Returns (analysis_text, key_factors, confidence_label) in the requested language."""

        cache_key = f"{prediction.match_id}:{language}"
        if cache_key in _ai_cache:
            logger.debug(f"AI cache hit: {cache_key}")
            return _ai_cache[cache_key]

        lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["en"])
        fmt_label = _FORMAT_LABELS.get(fmt, "Cricket")

        # Toss info
        toss_info = ""
        if prediction.toss_winner if hasattr(prediction, 'toss_winner') else False:
            pass  # toss data comes through venue_notes if available

        prompt = f"""You are an expert cricket analyst and tipster. {lang_instruction}

Analyze this {fmt_label} match and provide sharp, data-driven insights.

MATCH: {team1} vs {team2}
SERIES: {series}
FORMAT: {fmt_label}
START: {prediction.match_start_utc.strftime('%d %b %Y %H:%M UTC')}
{f'VENUE: {prediction.venue_name}' if prediction.venue_name else ''}
{f'PITCH: {prediction.pitch_type}' if prediction.pitch_type else ''}
{f'DEW FACTOR: Yes — favours chasing team' if prediction.dew_factor else ''}
{f'NOTES: {venue_notes}' if venue_notes else ''}

{team1.upper()} FORM:
- Last 5: {' '.join(team1_form.last_5) if team1_form.last_5 else 'N/A'} (W=Win, L=Loss, N=No Result)
- Record (W/L): {team1_form.wins}/{team1_form.losses} in {team1_form.matches_played} matches
- Avg runs scored: {team1_form.avg_runs_scored:.1f} | Avg conceded: {team1_form.avg_runs_conceded:.1f}

{team2.upper()} FORM:
- Last 5: {' '.join(team2_form.last_5) if team2_form.last_5 else 'N/A'}
- Record (W/L): {team2_form.wins}/{team2_form.losses} in {team2_form.matches_played} matches
- Avg runs scored: {team2_form.avg_runs_scored:.1f} | Avg conceded: {team2_form.avg_runs_conceded:.1f}

HEAD-TO-HEAD (last {h2h.matches_played} meetings):
- {team1} wins: {h2h.team1_wins} | {team2} wins: {h2h.team2_wins} | No result: {h2h.no_results}
- Recent H2H: {' '.join(h2h.recent_results) if h2h.recent_results else 'N/A'} (T1={team1}, T2={team2}, N=no result)

MODEL OUTPUT:
- {team1} win probability: {prediction.team1_win_prob*100:.1f}%
- {team2} win probability: {prediction.team2_win_prob*100:.1f}%
{f'- Draw probability: {prediction.draw_prob*100:.1f}%' if prediction.draw_prob > 0 else ''}
- Predicted first innings: {prediction.predicted_first_innings_score:.0f} runs ({prediction.predicted_first_innings_range[0]}-{prediction.predicted_first_innings_range[1]})
- Total runs O/U line: {prediction.total_runs_line} | Over prob: {prediction.over_line_prob*100:.1f}%
- Recommendation: {prediction.recommended_bet}
- Confidence: {prediction.confidence_label}

{'TOP BATSMEN (model picks):' if prediction.top_batsman_predictions else ''}
{chr(10).join(f'  {i+1}. {p.player_name} ({p.probability*100:.1f}%)' for i, p in enumerate(prediction.top_batsman_predictions[:3]))}

{'TOP BOWLERS (model picks):' if prediction.top_bowler_predictions else ''}
{chr(10).join(f'  {i+1}. {p.player_name} ({p.probability*100:.1f}%)' for i, p in enumerate(prediction.top_bowler_predictions[:3]))}

Provide your analysis in the following JSON format (all values in the requested language):
{{
  "analysis": "Exactly 4 lines. Line 1: current form summary. Line 2: key matchup or pitch factor. Line 3: why the model's pick makes sense. Line 4: main risk or caveat. Be sharp and specific — cite numbers.",
  "key_factors": ["factor 1 (max 10 words)", "factor 2", "factor 3"],
  "confidence": "Low|Medium|High",
  "value_assessment": "1 sentence max on betting value."
}}

Be concise. Every word must add value. Think like a sharp tipster, not a journalist."""

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0.3,
                system="You are an expert cricket analyst. Always respond with valid JSON only, no extra text.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            data = _extract_json(raw)

            analysis = data.get("analysis", "Analysis unavailable.")
            factors = data.get("key_factors", [])[:3]
            confidence = data.get("confidence", "Medium")
            value_note = data.get("value_assessment", "")
            if value_note:
                analysis = f"{analysis}\n\n**{('Value Assessment' if language == 'en' else 'मूल्य आकलन' if language == 'hi' else 'قدر کا جائزہ')}:** {value_note}"

            result = (analysis, factors, confidence)
            _ai_cache[cache_key] = result
            return result

        except json.JSONDecodeError:
            logger.warning(f"Claude returned non-JSON for cricket analysis ({language})")
            return "Analysis unavailable.", [], "Medium"
        except Exception as e:
            logger.error(f"Claude API error (cricket/{language}): {e}")
            return "AI analysis temporarily unavailable.", [], "Medium"
