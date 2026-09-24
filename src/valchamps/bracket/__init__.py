"""Monte Carlo bracket simulator (Phase 6)."""

from valchamps.bracket.event import fixed_results, group_openings
from valchamps.bracket.format import MatchSlot, build_bracket, load_format, validate
from valchamps.bracket.odds import elo_odds, model_odds
from valchamps.bracket.publish import publish, write_matchups
from valchamps.bracket.run import EventForecast, forecast_event, results_fingerprint
from valchamps.bracket.simulate import FixedResult, SimulationResult, simulate

__all__ = [
    "EventForecast",
    "FixedResult",
    "MatchSlot",
    "SimulationResult",
    "build_bracket",
    "elo_odds",
    "fixed_results",
    "forecast_event",
    "group_openings",
    "load_format",
    "model_odds",
    "publish",
    "results_fingerprint",
    "simulate",
    "validate",
    "write_matchups",
]
