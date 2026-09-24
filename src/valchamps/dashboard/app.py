"""Streamlit dashboard: title odds, how they moved, and a match predictor.

Run with ``valchamps dashboard`` (or ``streamlit run src/valchamps/dashboard/app.py``). Every
number comes from the API (``VALCHAMPS_API_URL``, default http://localhost:8000), or, when
``VALCHAMPS_DATA_URL`` is set, from the published ``odds/`` folder it points to (the hosted
version reads the repository on GitHub, so it needs no API or database).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from valchamps.dashboard.charts import PALETTE, history_lines, score_bars, title_bars
from valchamps.dashboard.client import (
    DEFAULT_API_URL,
    DEFAULT_DATA_URL,
    ApiClient,
    ApiError,
    StaticClient,
)

CHAMPIONS_2026 = 2766
REPO_URL = "https://github.com/JasSaini101/2026__Val_Champs_Model"

st.set_page_config(page_title="Champions 2026 odds", layout="wide")


def _theme() -> dict[str, str]:
    try:
        kind = st.context.theme.type
    except AttributeError:
        kind = None
    return PALETTE["dark" if kind == "dark" else "light"]


def _client(source: str) -> ApiClient | StaticClient:
    return StaticClient(source) if DEFAULT_DATA_URL else ApiClient(source)


@st.cache_data(ttl=60, show_spinner=False)
def _fetch(source: str, what: str, event: int):
    client = _client(source)
    return {"odds": client.odds, "history": client.history, "teams": client.teams}[what](event)


@st.cache_data(ttl=300, show_spinner="Simulating the veto and scoring every map...")
def _predict(source: str, a: int, b: int, best_of: int, event: int) -> dict:
    return _client(source).predict(a, b, best_of, event)


def _pct(x: float) -> str:
    return f"{x:.1%}"


with st.sidebar:
    if DEFAULT_DATA_URL:
        api_url = DEFAULT_DATA_URL
        st.caption("Reading the odds published by the hourly update job.")
    else:
        api_url = st.text_input("API URL", DEFAULT_API_URL)
    event = int(st.number_input("Event id", value=CHAMPIONS_2026, step=1))
    if st.button("Refresh"):
        st.cache_data.clear()
colors = _theme()

st.title("VALORANT Champions 2026")
st.markdown(
    "Each team's chance of winning Champions, from **100,000 simulated tournaments**. A model "
    "trained on 2025-26 pro matches gives the odds of each map, a simulated map veto turns those "
    "into Bo3/Bo5 odds, and the bracket is replayed from the results so far. It re-runs "
    f"whenever a match finishes. [Code and write-up]({REPO_URL})"
)
odds_tab, match_tab = st.tabs(["Title odds", "Match predictor"])

with odds_tab:
    try:
        latest = _fetch(api_url, "odds", event)
    except ApiError as exc:
        st.info(f"No published odds yet ({exc}). Run `valchamps update` to publish them.")
        latest = None
    if latest:
        teams = pd.DataFrame(latest["teams"])
        st.caption(
            f"Updated {latest.get('updated_at', '?')} · data through {latest['as_of'][:16]} · "
            f"{latest['fixed_results']} finished series · {latest['runs']:,} simulated "
            f"tournaments · odds from {latest['odds']}"
        )
        fav = teams.sort_values("title", ascending=False).iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Favourite", fav["team"])
        k2.metric("Favourite's title odds", _pct(fav["title"]))
        k3.metric("Finished series", latest["fixed_results"])
        k4.metric("Teams above 1% to win", int((teams["title"] >= 0.01).sum()))

        st.subheader("Chance to win the title")
        st.altair_chart(title_bars(teams, colors), width="stretch")
        with st.container():
            st.subheader("Every stage")
            table = teams.sort_values("title", ascending=False)[
                ["team", "group", "top8", "top4", "final", "title"]
            ]
            progress = lambda label: st.column_config.ProgressColumn(  # noqa: E731
                label, format="percent", min_value=0.0, max_value=1.0
            )
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height=36 * (len(table) + 1),
                column_config={
                    "team": "Team",
                    "group": "Group",
                    "top8": progress("Playoffs"),
                    "top4": progress("Top 4"),
                    "final": progress("Final"),
                    "title": progress("Title"),
                },
            )

        st.subheader("How the title odds moved")
        try:
            history = pd.DataFrame(_fetch(api_url, "history", event))
        except ApiError:
            history = pd.DataFrame()
        if history.empty or history["updated_at"].nunique() < 2:
            st.caption("The chart appears after the next published update (a new result).")
        else:
            names = teams.sort_values("title", ascending=False)["team"].tolist()
            highlight = st.selectbox("Highlight", names, index=0)
            st.altair_chart(history_lines(history, highlight, colors), width="stretch")

with match_tab:
    try:
        teams_list = _fetch(api_url, "teams", event)
    except ApiError as exc:
        st.error(str(exc))
        teams_list = []
    if teams_list:
        label = {t["team_id"]: f"{t['name']} ({t['group']})" for t in teams_list}
        ids = sorted(label, key=lambda t: label[t])
        c1, c2, c3 = st.columns([2, 2, 1])
        a = c1.selectbox("Team A", ids, format_func=label.get, index=0)
        b = c2.selectbox("Team B", ids, format_func=label.get, index=min(1, len(ids) - 1))
        best_of = c3.radio("Format", [3, 5], format_func=lambda n: f"Bo{n}", horizontal=True)
        if a == b:
            st.warning("Pick two different teams.")
        else:
            try:
                pred = _predict(api_url, a, b, best_of, event)
            except ApiError as exc:
                st.error(str(exc))
                pred = None
            if pred:
                name_a, name_b = pred["team_a"]["name"], pred["team_b"]["name"]
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{name_a} wins", _pct(pred["p_a"]))
                m2.metric(f"{name_b} wins", _pct(pred["p_b"]))
                m3.metric(
                    f"Raw Elo, {name_a}",
                    _pct(pred["elo_p_a"]),
                    help="Elo alone: no map or veto information",
                )
                left, right = st.columns([2, 3])
                with left:
                    st.subheader("Final score")
                    st.altair_chart(
                        score_bars(pred["scores"], name_a, name_b, colors), width="stretch"
                    )
                with right:
                    st.subheader(f"{name_a}'s chance on each map")
                    maps = pd.DataFrame(pred["maps"])
                    pct = lambda lbl: st.column_config.NumberColumn(lbl, format="percent")  # noqa: E731
                    st.dataframe(
                        maps,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "map": "Map",
                            "pick_a": pct(f"{name_a} pick"),
                            "pick_b": pct(f"{name_b} pick"),
                            "decider": pct("Decider"),
                            "in_series": st.column_config.ProgressColumn(
                                "In the series", format="percent", min_value=0.0, max_value=1.0
                            ),
                        },
                    )
                st.subheader("Likeliest vetoes")
                who = {"pick_a": name_a, "pick_b": name_b, "decider": "decider"}
                route = lambda v: " → ".join(  # noqa: E731
                    f"{m['map']} ({who[m['picked_by']]})" for m in v["maps"]
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"probability": v["p"], "vetoes": route(v)} for v in pred["vetoes"]]
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "probability": st.column_config.NumberColumn(
                            "Probability", format="percent"
                        ),
                        "vetoes": "Maps in play order (who picked)",
                    },
                )
                st.caption(
                    f"Map pool {', '.join(pred['map_pool'])} · model {pred['model']['name']} "
                    f"trained through {pred['model']['trained_through']} · data through "
                    f"{pred['data_through'][:16]}"
                )
