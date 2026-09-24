"""A tournament as an ordered list of matches whose slots point at earlier matches.

A slot is ``T:<team_id>`` (a team placed there from the start), ``W:<match>`` or ``L:<match>``
(the winner or loser of an earlier match), or a group seed like ``A1`` (group A's winner),
resolved to the group match that decides it. Every team leaves the bracket through a
``loser_place`` (or the final's ``winner_place``), which is its final standing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REF = re.compile(r"^(W|L):(.+)$")


@dataclass(frozen=True)
class MatchSlot:
    id: str
    stage: str  # vlr.gg's stage label, used to fix finished matches to their real result
    a: str
    b: str
    best_of: int = 3
    loser_place: str | None = None
    winner_place: str | None = None


def load_format(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def gsl_group(
    name: str, openings: tuple[tuple[int, int], tuple[int, int]], best_of: int,
    elimination_place: str, decider_place: str,
) -> tuple[list[MatchSlot], dict[str, str]]:  # fmt: skip
    """A four-team GSL group and its seeds: 1st wins the Winner's match, 2nd the Decider."""
    (a, b), (c, d) = openings
    g = f"{name}-"
    stage = lambda s: f"Group Stage: {s} ({name})"  # noqa: E731
    matches = [
        MatchSlot(f"{g}O1", stage("Opening"), f"T:{a}", f"T:{b}", best_of),
        MatchSlot(f"{g}O2", stage("Opening"), f"T:{c}", f"T:{d}", best_of),
        MatchSlot(f"{g}W", stage("Winner's"), f"W:{g}O1", f"W:{g}O2", best_of),
        MatchSlot(f"{g}E", stage("Elimination"), f"L:{g}O1", f"L:{g}O2", best_of,
                  loser_place=elimination_place),
        MatchSlot(f"{g}D", stage("Decider"), f"L:{g}W", f"W:{g}E", best_of,
                  loser_place=decider_place),
    ]  # fmt: skip
    return matches, {f"{name}1": f"W:{g}W", f"{name}2": f"W:{g}D"}


def build_bracket(
    config: dict[str, Any], openings: dict[str, tuple[tuple[int, int], tuple[int, int]]]
) -> list[MatchSlot]:
    """Group stage then playoffs, in an order where every slot refers to an earlier match.

    ``openings`` maps each group name to its two opening pairs of team ids.
    """
    groups = config["groups"]
    missing = [g for g in groups["names"] if g not in openings]
    if missing:
        raise ValueError(f"no opening matches for group(s) {missing}")
    matches: list[MatchSlot] = []
    seeds: dict[str, str] = {}
    for name in groups["names"]:
        group_matches, group_seeds = gsl_group(
            name, openings[name], groups.get("best_of", 3),
            groups["elimination_place"], groups["decider_place"],
        )  # fmt: skip
        matches += group_matches
        seeds |= group_seeds
    for m in config.get("playoffs", []):
        matches.append(
            MatchSlot(
                id=m["id"],
                stage=m["stage"],
                a=seeds.get(m["a"], m["a"]),
                b=seeds.get(m["b"], m["b"]),
                best_of=m.get("best_of", 3),
                loser_place=m.get("loser_place"),
                winner_place=m.get("winner_place"),
            )
        )
    validate(matches)
    return matches


def validate(matches: list[MatchSlot]) -> None:
    """Every slot is a team or an earlier result, and every result is used exactly once."""
    seen: set[str] = set()
    used: dict[str, int] = {}
    for m in matches:
        if m.id in seen:
            raise ValueError(f"duplicate match id {m.id}")
        for slot in (m.a, m.b):
            ref = _REF.match(slot)
            if ref:
                if ref.group(2) not in seen:
                    raise ValueError(f"{m.id} refers to {slot} before that match")
                used[slot] = used.get(slot, 0) + 1
            elif not slot.startswith("T:"):
                raise ValueError(f"{m.id}: unknown slot {slot!r}")
        seen.add(m.id)
    for m in matches:
        for kind, place in (("W", m.winner_place), ("L", m.loser_place)):
            n = used.get(f"{kind}:{m.id}", 0) + (place is not None)
            if n != 1:
                what = "winner" if kind == "W" else "loser"
                raise ValueError(f"the {what} of {m.id} goes to {n} places; it needs exactly 1")


def place_order(label: str) -> int:
    """Sort key for standings like "1", "5-6", "13-16"."""
    return int(label.split("-")[0])
