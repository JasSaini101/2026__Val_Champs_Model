"""Series model (Phase 5): simulated vetoes and exact Bo3/Bo5 odds from per-map odds."""

from valchamps.series.backtest import (
    METHODS,
    SeriesResult,
    actual_veto,
    format_series_table,
    run_series_backtest,
    veto_summary,
)
from valchamps.series.params import SeriesParams
from valchamps.series.predict import (
    SeriesPrediction,
    map_probabilities,
    predict_matchups,
    predict_series,
    prediction_payload,
)
from valchamps.series.series import (
    score_distribution,
    series_win_prob,
    veto_score_distribution,
    veto_win_prob,
)
from valchamps.series.veto import (
    FORMATS,
    VetoOutcome,
    VetoTendencies,
    map_play_probabilities,
    simulate_veto,
    tendencies,
)

__all__ = [
    "FORMATS",
    "METHODS",
    "SeriesParams",
    "SeriesPrediction",
    "SeriesResult",
    "VetoOutcome",
    "VetoTendencies",
    "actual_veto",
    "format_series_table",
    "map_play_probabilities",
    "map_probabilities",
    "predict_matchups",
    "predict_series",
    "prediction_payload",
    "run_series_backtest",
    "score_distribution",
    "series_win_prob",
    "simulate_veto",
    "tendencies",
    "veto_score_distribution",
    "veto_summary",
    "veto_win_prob",
]
