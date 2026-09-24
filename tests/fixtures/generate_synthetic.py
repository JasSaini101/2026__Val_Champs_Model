"""Regenerate the synthetic vlr.gg-style fixtures in ``tests/fixtures/vlr/``.

Markup mirrors vlr.gg; every number and id is made up. Run from the repo root:
    uv run python tests/fixtures/generate_synthetic.py
"""

from pathlib import Path

OUT = Path(__file__).parent / "vlr"
BANNER = "<!-- SYNTHETIC FIXTURE: markup mirrors vlr.gg, all stats/ids are illustrative, not real match data. -->\n"

FNC = [(9001, "Boaster"), (9002, "Chronicle"), (9003, "Derke"), (9004, "Alfajer"), (9005, "Leo")]
TH = [(9101, "Boo"), (9102, "benjyfishy"), (9103, "MiniBoo"), (9104, "RieNs"), (9105, "Wo0t")]
AGENTS = ["Astra", "Viper", "Raze", "Killjoy", "Sova"]


def stat(v):
    return f'<td class="mod-stat"><span class="stats-sq"><span class="side mod-side mod-both">{v}</span><span class="side mod-side mod-t">{v}</span><span class="side mod-side mod-ct">{v}</span></span></td>'


def deaths(v):
    return f'<td class="mod-stat mod-vlr-deaths"><span class="stats-sq"><span class="num-sep">/</span><span class="side mod-both">{v}</span></span></td>'


def table(players, tag, seed):
    rows = []
    for i, (pid, handle) in enumerate(players):
        k = 12 + (seed + i * 3) % 11
        d = 10 + (seed + i * 5) % 9
        a = 3 + (seed + i) % 7
        cells = [
            stat(f"{0.8 + ((seed + i * 7) % 60) / 100:.2f}"),
            stat(150 + (seed * 13 + i * 29) % 140),
            f'<td class="mod-stat mod-vlr-kills"><span class="stats-sq"><span class="side mod-both">{k}</span></span></td>',
            deaths(d),
            stat(a),
            stat(f"{k - d:+d}"),
            stat(f"{65 + (seed + i * 4) % 25}%"),
            stat(100 + (seed * 7 + i * 17) % 90),
            stat(f"{18 + (seed + i * 3) % 20}%"),
            stat((seed + i) % 5),
            stat((seed + i * 2) % 4),
            stat(f"{((seed + i) % 5) - ((seed + i * 2) % 4):+d}"),
        ]
        rows.append(
            f'<tr><td class="mod-player"><div><a href="/player/{pid}/{handle.lower()}">'
            f'<div class="text-of">{handle}</div><div class="ge-text-light">{tag}</div></a></div></td>'
            f'<td class="mod-agents"><div><span class="stats-sq mod-agent small">'
            f'<img src="/img/vlr/game/agents/{AGENTS[i].lower()}.png" alt="{AGENTS[i].lower()}" title="{AGENTS[i]}"></span></div></td>'
            + "".join(cells)
            + "</tr>"
        )
    head = "<thead><tr><th></th><th></th><th>R</th><th>ACS</th><th>K</th><th>D</th><th>A</th><th>+/–</th><th>KAST</th><th>ADR</th><th>HS%</th><th>FK</th><th>FD</th><th>+/–</th></tr></thead>"
    return (
        f'<table class="wf-table-inset mod-overview">{head}<tbody>{"".join(rows)}</tbody></table>'
    )


def side(name, score, win, ct, t, right=False, ot=None):
    score_div = f'<div class="score{" mod-win" if win else ""}">{score}</div>'
    halves = f'<span class="mod-ct">{ct}</span> / <span class="mod-t">{t}</span>' + (
        f' / <span class="mod-ot">{ot}</span>' if ot is not None else ""
    )
    inner = f'<div><div class="team-name">{name}</div>{halves}</div>'
    body = inner + score_div if right else score_div + inner
    return f'<div class="team{" mod-right" if right else ""}">{body}</div>'


CT_OUTCOMES = ["elim", "defuse", "elim", "time"]
T_OUTCOMES = ["elim", "boom", "elim"]


def round_winners(t1, t2):
    """Round-by-round (winner, winner_side) consistent with the half scores.

    Team 1 starts on defence (CT); sides swap after 12 rounds and every overtime round.
    """
    (_, ct1, tt1, ot1), (_, ct2, tt2, ot2) = t1, t2
    first = [1] * ct1 + [2] * tt2
    second = [1] * tt1 + [2] * ct2
    order = lambda n: sorted(range(n), key=lambda i: (i * 7) % 13)  # noqa: E731
    first = [first[i] for i in order(len(first))]
    second = [second[i] for i in order(len(second))]
    rounds = [(w, "ct" if w == 1 else "t") for w in first]
    rounds += [(w, "t" if w == 1 else "ct") for w in second]
    for n in range((ot1 or 0) + (ot2 or 0)):
        w = 1 if n < (ot1 or 0) else 2
        team1_ct = n % 2 == 0
        rounds.append((w, "ct" if (w == 1) == team1_ct else "t"))
    return rounds


def rounds_block(t1, t2):
    cols = [
        '<div class="vlr-rounds-row-col"><div style="height: 12px;"></div>'
        '<div class="team"><img src="//owcdn.net/img/fnc.png"> FNC</div>'
        '<div class="team"><img src="//owcdn.net/img/th.png"> TH</div></div>'
    ]
    s1 = s2 = 0
    for i, (w, side) in enumerate(round_winners(t1, t2), start=1):
        s1, s2 = (s1 + 1, s2) if w == 1 else (s1, s2 + 1)
        pool = CT_OUTCOMES if side == "ct" else T_OUTCOMES
        win = (
            f'<div class="rnd-sq mod-win mod-{side}">'
            f'<img src="/img/vlr/game/round/{pool[i % len(pool)]}.webp"></div>'
        )
        empty = '<div class="rnd-sq"></div>'
        cells = win + empty if w == 1 else empty + win
        cols.append(
            f'<div class="vlr-rounds-row-col" title="{s1}-{s2}">'
            f'<div class="rnd-num">{i}</div>{cells}</div>'
        )
        if i == 12:
            cols.append('<div class="vlr-rounds-row-col mod-spacing"></div>')
    return (
        '<div style="text-align: center; margin-top: 15px;">'
        '<div style="overflow-x: auto; text-align: center;"><div class="vlr-rounds">'
        f'<div class="vlr-rounds-row">{"".join(cols)}</div></div></div></div>'
    )


def game(gid, map_name, pick, t1, t2, dur, seed, with_tables=True):
    pick_html = f' <span class="picked mod-{pick} ge-text-light">PICK</span>' if pick else ""
    (s1, ct1, tt1, ot1), (s2, ct2, tt2, ot2) = t1, t2
    header = (
        '<div class="vm-stats-game-header">'
        + side("FNATIC", s1, s1 > s2, ct1, tt1, ot=ot1)
        + f'<div class="map"><div style="font-weight: 700; font-size: 20px;"><span style="position: relative;">{map_name}{pick_html}</span></div>'
        + f'<div class="map-duration ge-text-light">{dur}</div></div>'
        + side("Team Heretics", s2, s2 > s1, ct2, tt2, right=True, ot=ot2)
        + "</div>"
    )
    tables = (
        (
            f'<div style="overflow-x: auto;">{table(FNC, "FNC", seed)}</div>'
            f'<div style="overflow-x: auto;">{table(TH, "TH", seed + 1)}</div>'
        )
        if with_tables
        else ""
    )
    return f'<div class="vm-stats-game" data-game-id="{gid}">{header}{rounds_block(t1, t2)}{tables}</div>'


def completed_match(with_tables=True):
    return (
        BANNER
        + f"""<!DOCTYPE html><html><head><title>FNATIC vs. Team Heretics | VLR.gg</title></head><body>
<div class="wf-card match-header">
  <div class="match-header-super">
    <div><a href="/event/2097/valorant-champions-2024/playoffs" class="match-header-event">
      <div><div style="font-weight: 700;">Valorant Champions 2024</div>
      <div class="match-header-event-series">Playoffs: Upper Final</div></div></a></div>
    <div class="match-header-date">
      <div class="moment-tz-convert" data-utc-ts="2024-08-22 12:00:00" data-moment-format="dddd, MMMM Do">Thursday, August 22nd</div>
      <div class="moment-tz-convert" data-utc-ts="2024-08-22 12:00:00" data-moment-format="h:mm A z">12:00 PM UTC</div>
      <div style="margin-top: 4px;"><div class="wf-tooltip">Patch 9.03</div></div>
    </div>
  </div>
  <div class="match-header-vs">
    <a class="match-header-link wf-link-hover mod-1" href="/team/2593/fnatic">
      <div class="match-header-link-name mod-1"><div class="wf-title-med">FNATIC</div><div class="match-header-link-name-elo">[1923]</div></div></a>
    <div class="match-header-vs-score">
      <div class="match-header-vs-note"><span class="match-header-vs-note mod-upcoming">final</span></div>
      <div><div class="js-spoiler"><span class="match-header-vs-score-winner">2</span><span class="match-header-vs-score-colon">:</span><span class="match-header-vs-score-loser">1</span></div></div>
      <div class="match-header-vs-note">Bo3</div>
    </div>
    <a class="match-header-link wf-link-hover mod-2" href="/team/1001/team-heretics">
      <div class="match-header-link-name mod-2"><div class="wf-title-med">Team Heretics</div></div></a>
  </div>
  <div class="match-header-note">FNC ban Icebox; TH ban Sunset; FNC pick Lotus; TH pick Split; FNC ban Bind; TH ban Haven; Abyss remains</div>
</div>
<div class="vm-stats">
  <div class="vm-stats-gamesnav"><div class="vm-stats-gamesnav-item js-map-switch" data-game-id="all">All Maps</div></div>
  <div class="vm-stats-container">
    <div class="vm-stats-game mod-active" data-game-id="all">
      {table(FNC, "FNC", 50) if with_tables else ""}
      {table(TH, "TH", 51) if with_tables else ""}
    </div>
    {game(180001, "Lotus", 1, (13, 7, 6, None), (10, 5, 5, None), "49:31", 3, with_tables)}
    {game(180002, "Split", 2, (7, 3, 4, None), (13, 4, 9, None), "38:02", 7, with_tables)}
    {game(180003, "Abyss", None, (14, 6, 6, 2), (12, 6, 6, 0), "1:02:47", 11, with_tables)}
  </div>
</div>
</body></html>
"""
    )


match_upcoming = (
    BANNER
    + """<!DOCTYPE html><html><head><title>EDward Gaming vs. FNATIC | VLR.gg</title></head><body>
<div class="wf-card match-header">
  <div class="match-header-super">
    <div><a href="/event/2097/valorant-champions-2024/playoffs" class="match-header-event">
      <div><div style="font-weight: 700;">Valorant Champions 2024</div>
      <div class="match-header-event-series">Playoffs: Grand Final</div></div></a></div>
    <div class="match-header-date">
      <div class="moment-tz-convert" data-utc-ts="2024-08-25 10:00:00" data-moment-format="dddd, MMMM Do">Sunday, August 25th</div>
    </div>
  </div>
  <div class="match-header-vs">
    <a class="match-header-link wf-link-hover mod-1" href="/team/1120/edward-gaming">
      <div class="match-header-link-name mod-1"><div class="wf-title-med">EDward Gaming</div></div></a>
    <div class="match-header-vs-score">
      <div class="match-header-vs-note"><span class="match-header-vs-note mod-upcoming">upcoming</span></div>
      <div class="match-header-vs-placeholder">vs.</div>
      <div class="match-header-vs-note">Bo5</div>
    </div>
    <a class="match-header-link wf-link-hover mod-2" href="/team/2593/fnatic">
      <div class="match-header-link-name mod-2"><div class="wf-title-med">FNATIC</div></div></a>
  </div>
</div>
<div class="vm-stats"><div class="vm-stats-container"></div></div>
</body></html>
"""
)


def card(mid, slug, t1, t2, s1, s2, status, series):
    return f"""<a href="/{mid}/{slug}" class="wf-module-item match-item mod-color mod-left mod-bg-after-striped_purple">
  <div class="match-item-time">12:00 PM</div>
  <div class="match-item-vs">
    <div class="match-item-vs-team"><div class="match-item-vs-team-name"><div class="text-of"><span class="flag mod-eu"></span> {t1}</div></div><div class="match-item-vs-team-score js-spoiler">{s1}</div></div>
    <div class="match-item-vs-team"><div class="match-item-vs-team-name"><div class="text-of"><span class="flag mod-cn"></span> {t2}</div></div><div class="match-item-vs-team-score js-spoiler">{s2}</div></div>
  </div>
  <div class="match-item-eta"><div class="ml"><div class="ml-status">{status}</div></div></div>
  <div class="match-item-event text-of"><div class="match-item-event-series text-of">{series}</div> Valorant Champions 2024</div>
</a>"""


event_matches = (
    BANNER
    + f"""<!DOCTYPE html><html><head><title>Valorant Champions 2024: Matches | VLR.gg</title></head><body>
<div class="wf-card mod-event mod-header mod-full"><h1 class="wf-title">Valorant Champions 2024</h1></div>
<div class="wf-label mod-large">Thu, August 22, 2024</div>
<div class="wf-card">
{card(378829, "fnatic-vs-team-heretics-valorant-champions-2024-ubf", "FNATIC", "Team Heretics", 2, 1, "Completed", "Playoffs–Upper Final")}
</div>
<div class="wf-label mod-large">Sat, August 24, 2024</div>
<div class="wf-card">
{card(378831, "team-heretics-vs-edward-gaming-valorant-champions-2024-lf", "Team Heretics", "EDward Gaming", "", "", "LIVE", "Playoffs–Lower Final")}
</div>
<div class="wf-label mod-large">Sun, August 25, 2024</div>
<div class="wf-card">
{card(378830, "edward-gaming-vs-fnatic-valorant-champions-2024-gf", "EDward Gaming", "FNATIC", "–", "–", "Upcoming", "Playoffs–Grand Final")}
<a href="/378999/tbd-vs-tbd" class="wf-module-item match-item"><div class="match-item-vs"><div class="match-item-vs-team"><div class="match-item-vs-team-name"><div class="text-of">TBD</div></div></div></div></a>
</div>
</body></html>
"""
)

event_page = (
    BANNER
    + """<!DOCTYPE html><html><head><title>Valorant Champions 2024 | VLR.gg</title></head><body>
<div class="wf-card mod-event mod-header mod-full">
  <div class="event-header"><div class="event-desc"><div class="event-desc-inner">
    <h1 class="wf-title">Valorant Champions 2024</h1>
    <h2 class="event-desc-subtitle">Seoul &amp; Incheon, South Korea</h2>
    <div class="event-desc-items">
      <div class="event-desc-item"><div class="event-desc-item-label">Dates</div><div class="event-desc-item-value">Aug 1, 2024 - Aug 25, 2024</div></div>
      <div class="event-desc-item"><div class="event-desc-item-label">Prize pool</div><div class="event-desc-item-value">$2,250,000 USD</div></div>
      <div class="event-desc-item"><div class="event-desc-item-label">Location</div><div class="event-desc-item-value">Seoul &amp; Incheon, South Korea</div></div>
    </div>
  </div></div></div>
</div>
</body></html>
"""
)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "match_378829_completed.html").write_text(completed_match())
# What vlr.gg actually served in Sep 2026: round-by-round block but no player stats tables.
(OUT / "match_378829_no_stats.html").write_text(completed_match(with_tables=False))
(OUT / "match_378830_upcoming.html").write_text(match_upcoming)
(OUT / "event_matches_2097.html").write_text(event_matches)
(OUT / "event_2097.html").write_text(event_page)
