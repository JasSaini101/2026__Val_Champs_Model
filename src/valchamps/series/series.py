"""Exact best-of-N odds from per-map win probabilities.

Maps are treated as independent given their probabilities. A series stops once a team has won
a majority, but that never changes who wins, only the final score.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from valchamps.series.veto import VetoOutcome

MapProbs = Mapping[tuple[str, str], float]  # (map, context) -> P(team A wins that map)


def score_distribution(ps: Sequence[float]) -> dict[tuple[int, int], float]:
    """P(final score) for a series on maps won by team A with probabilities ``ps``, in order.

    Keys are (maps won by A, maps won by B), e.g. (2, 1).
    """
    if len(ps) % 2 == 0:
        raise ValueError(f"a series needs an odd number of maps, got {len(ps)}")
    need = len(ps) // 2 + 1
    live: dict[tuple[int, int], float] = {(0, 0): 1.0}
    final: defaultdict[tuple[int, int], float] = defaultdict(float)
    for p in ps:
        nxt: defaultdict[tuple[int, int], float] = defaultdict(float)
        for (won, lost), q in live.items():
            for score, r in (((won + 1, lost), q * p), ((won, lost + 1), q * (1 - p))):
                (final if need in score else nxt)[score] += r
        live = nxt
    return dict(final)


def series_win_prob(ps: Sequence[float]) -> float:
    """P(team A wins the series) given its win probability on each map, in play order."""
    need = len(ps) // 2 + 1
    return sum(q for (won, _), q in score_distribution(ps).items() if won == need)


def veto_win_prob(outcomes: Iterable[VetoOutcome], map_probs: MapProbs) -> float:
    """P(team A wins), averaged over the possible vetoes."""
    return sum(o.prob * series_win_prob([map_probs[m] for m in o.maps]) for o in outcomes)


def veto_score_distribution(
    outcomes: Iterable[VetoOutcome], map_probs: MapProbs
) -> dict[tuple[int, int], float]:
    out: defaultdict[tuple[int, int], float] = defaultdict(float)
    for o in outcomes:
        for score, q in score_distribution([map_probs[m] for m in o.maps]).items():
            out[score] += o.prob * q
    return dict(out)
