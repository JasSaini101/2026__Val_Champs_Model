"""Simulated map vetoes in the VCT formats.

Seven maps are in the pool and the two teams take turns, the team that vetoes first taking
steps 1, 3, 5 (and 7 in a Bo5 pick phase)::

    Bo3: ban, ban, pick, pick, ban, ban, decider
    Bo5: ban, ban, pick, pick, pick, pick, decider

At each step the acting team chooses among the maps still available in proportion to how often
it has banned (or picked) each of them before, plus ``prior`` pseudo-vetoes per map. With no
history (or a large prior) that is uniform. Which team vetoes first is a coin flip.

Every possible veto is enumerated exactly (at most 7! sequences per starting team), collapsed to
what matters for the series: the maps in play order and who picked each.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

from valchamps.features.map_pool import MapPoolTracker

BAN, PICK = "ban", "pick"
FORMATS: dict[int, tuple[str, ...]] = {
    3: (BAN, BAN, PICK, PICK, BAN, BAN),
    5: (BAN, BAN, PICK, PICK, PICK, PICK),
}  # the map left after these steps is the decider

# A series' maps in play order, each with how it was chosen from team A's point of view:
# "pick_a", "pick_b" or "decider" (the contexts of FeatureBuilder.hypothetical_rows).
MapSequence = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class VetoTendencies:
    """How many times a team has picked and banned each map."""

    picks: Mapping[str, float] = field(default_factory=dict)
    bans: Mapping[str, float] = field(default_factory=dict)

    def weights(self, action: str, maps: Sequence[str], prior: float) -> list[float]:
        counts = self.picks if action == PICK else self.bans
        w = [counts.get(m, 0.0) + prior for m in maps]
        return w if sum(w) > 0 else [1.0] * len(maps)


def tendencies(tracker: MapPoolTracker, team: int) -> VetoTendencies:
    """``team``'s pick and ban counts so far, read from the map-pool tracker."""
    return VetoTendencies(
        picks={m: n for (t, m), n in tracker.picks.items() if t == team},
        bans={m: n for (t, m), n in tracker.bans.items() if t == team},
    )


@dataclass(frozen=True)
class VetoOutcome:
    maps: MapSequence
    prob: float


def simulate_veto(
    pool: Iterable[str], best_of: int, team_a: VetoTendencies, team_b: VetoTendencies,
    prior: float,
) -> list[VetoOutcome]:  # fmt: skip
    """Every distinct outcome of the veto with its probability (they sum to 1), likeliest first."""
    if best_of not in FORMATS:
        raise ValueError(f"no veto format for best-of-{best_of}; known: {sorted(FORMATS)}")
    steps = FORMATS[best_of]
    pool = frozenset(pool)
    if len(pool) != len(steps) + 1:
        raise ValueError(f"a best-of-{best_of} veto needs {len(steps) + 1} maps, got {len(pool)}")
    total: defaultdict[MapSequence, float] = defaultdict(float)
    for first, second, labels in ((team_a, team_b, ("pick_a", "pick_b")),
                                  (team_b, team_a, ("pick_b", "pick_a"))):  # fmt: skip
        for maps, prob in _enumerate(steps, (first, second), labels, prior, pool).items():
            total[maps] += 0.5 * prob
    return sorted((VetoOutcome(maps, prob) for maps, prob in total.items()), key=lambda o: -o.prob)


def _enumerate(
    steps: tuple[str, ...], teams: tuple[VetoTendencies, VetoTendencies],
    labels: tuple[str, str], prior: float, pool: frozenset[str],
) -> dict[MapSequence, float]:  # fmt: skip
    """Distribution of the map sequence when ``teams[0]`` vetoes first."""

    @cache
    def rest(step: int, remaining: frozenset[str]) -> dict[MapSequence, float]:
        if step == len(steps):
            (decider,) = remaining
            return {((decider, "decider"),): 1.0}
        turn = step % 2
        options = sorted(remaining)
        weights = teams[turn].weights(steps[step], options, prior)
        norm = sum(weights)
        out: defaultdict[MapSequence, float] = defaultdict(float)
        for map_name, w in zip(options, weights, strict=True):
            if w == 0:
                continue
            head = ((map_name, labels[turn]),) if steps[step] == PICK else ()
            for tail, p in rest(step + 1, remaining - {map_name}).items():
                out[head + tail] += w / norm * p
        return out

    return rest(0, pool)


def map_play_probabilities(outcomes: Iterable[VetoOutcome]) -> dict[str, float]:
    """Chance each map is in the series' map list (played if the series goes the distance)."""
    out: defaultdict[str, float] = defaultdict(float)
    for o in outcomes:
        for map_name, _ in o.maps:
            out[map_name] += o.prob
    return dict(out)
