"""Altair charts for the dashboard.

Colors come from a validated palette (checked for colour-vision deficiency in light and dark
mode): one sequential blue for magnitude, and blue/orange for the two teams of a match.
Everything else is recessive gray. Each chart has tooltips; the table views next to them carry
the exact numbers.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

PALETTE = {
    "light": {
        "accent": "#2a78d6",
        "team_a": "#2a78d6",
        "team_b": "#eb6834",
        "context": "#c3c2b7",
        "grid": "#e8e7e2",
        "text": "#52514e",
    },
    "dark": {
        "accent": "#3987e5",
        "team_a": "#3987e5",
        "team_b": "#d95926",
        "context": "#52514e",
        "text": "#c3c2b7",
        "grid": "#2e2e2c",
    },
}

PCT = ".1%"


def _axes(chart: alt.Chart, colors: dict[str, str]) -> alt.Chart:
    return chart.configure_axis(
        gridColor=colors["grid"],
        domain=False,
        tickColor=colors["grid"],
        labelFontSize=12,
        titleFontSize=12,
        titleFontWeight="normal",
    ).configure_view(stroke=None)


def title_bars(teams: pd.DataFrame, colors: dict[str, str]) -> alt.Chart:
    """Title odds per team, highest first: one sequential hue, value labels at the bar ends."""
    df = teams.sort_values("title", ascending=False)
    order = df["team"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("team:N", sort=order, title=None, axis=alt.Axis(labelLimit=180)),
        x=alt.X("title:Q", title="P(win the title)", axis=alt.Axis(format="%", tickCount=5)),
        tooltip=[
            alt.Tooltip("team:N", title="Team"),
            alt.Tooltip("title:Q", title="Title", format=PCT),
            alt.Tooltip("final:Q", title="Reach final", format=PCT),
            alt.Tooltip("top4:Q", title="Top 4", format=PCT),
            alt.Tooltip("top8:Q", title="Playoffs", format=PCT),
        ],
    )
    bars = base.mark_bar(color=colors["accent"], cornerRadiusEnd=4, height={"band": 0.7})
    labels = base.mark_text(align="left", dx=4, fontSize=11, color=colors["text"]).encode(
        text=alt.Text("title:Q", format=PCT)
    )
    return _axes((bars + labels).properties(height=28 * len(df)), colors)


def history_lines(history: pd.DataFrame, highlight: str, colors: dict[str, str]) -> alt.Chart:
    """Title odds over the published updates: every team in gray, one team in the accent."""
    df = history.assign(updated_at=pd.to_datetime(history["updated_at"]))
    tooltip = [
        alt.Tooltip("team:N", title="Team"),
        alt.Tooltip("updated_at:T", title="Updated", format="%b %d %H:%M"),
        alt.Tooltip("title:Q", title="Title", format=PCT),
        alt.Tooltip("fixed_results:Q", title="Results in"),
    ]
    x = alt.X("updated_at:T", title=None)  # Vega picks the tick format for the time span
    y = alt.Y("title:Q", title="P(win the title)", axis=alt.Axis(format="%", tickCount=5))
    others = (
        alt.Chart(df[df["team"] != highlight])
        .mark_line(strokeWidth=1, color=colors["context"])
        .encode(x=x, y=y, detail="team:N", tooltip=tooltip)
    )
    chosen = df[df["team"] == highlight]
    line = alt.Chart(chosen).mark_line(strokeWidth=2, color=colors["accent"]).encode(x=x, y=y)
    points = (
        alt.Chart(chosen)
        .mark_point(filled=True, size=64, color=colors["accent"])
        .encode(x=x, y=y, tooltip=tooltip)
    )
    return _axes((others + line + points).properties(height=320), colors)


def score_bars(scores: list[dict], name_a: str, name_b: str, colors: dict[str, str]) -> alt.Chart:
    """Final-score distribution, colored by which team wins the series (legend + labels)."""
    df = pd.DataFrame(scores)
    df["score"] = df["a"].astype(str) + "-" + df["b"].astype(str)
    df["winner"] = [name_a if a > b else name_b for a, b in zip(df["a"], df["b"], strict=True)]
    color = alt.Color(
        "winner:N",
        title="Series winner",
        scale=alt.Scale(domain=[name_a, name_b], range=[colors["team_a"], colors["team_b"]]),
        legend=alt.Legend(orient="top"),
    )
    base = alt.Chart(df).encode(
        x=alt.X(
            "score:N",
            sort=df["score"].tolist(),
            title=f"Final score ({name_a}-{name_b})",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("p:Q", title="Probability", axis=alt.Axis(format="%", tickCount=4)),
        tooltip=[alt.Tooltip("score:N", title="Score"), alt.Tooltip("p:Q", title="P", format=PCT)],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, width={"band": 0.6}).encode(color=color)
    labels = base.mark_text(dy=-6, fontSize=11, color=colors["text"]).encode(
        text=alt.Text("p:Q", format=PCT)
    )
    return _axes((bars + labels).properties(height=260), colors)
