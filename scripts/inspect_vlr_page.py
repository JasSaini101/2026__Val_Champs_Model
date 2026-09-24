"""Show how a vlr.gg match page is structured, to debug parser selectors.

Usage:
    uv run python scripts/inspect_vlr_page.py                     # first cached match page
    uv run python scripts/inspect_vlr_page.py path/to/page.html   # a specific saved page
    uv run python scripts/inspect_vlr_page.py --fetch "/378829/some-match/?game=all&tab=overview"
        # fetch a URL path from vlr.gg (rate-limited, cached like the scraper) and inspect it
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from valchamps.config import Settings
from valchamps.data.parser import _player_rows, parse_match
from valchamps.data.scraper import VlrClient

MARKERS = ("mod-overview", "wf-table", "mod-player", "stats-sq", "/player/", "vlr-rounds")


def load(args: list[str]) -> tuple[str, str]:
    if args[:1] == ["--fetch"]:
        path = args[1]
        with VlrClient() as client:
            return path, client.get(path, max_age=0)
    if args:
        page = Path(args[0])
    else:
        pages = sorted(p for p in Settings().cache_dir.glob("*.html") if p.name[0].isdigit())
        if not pages:
            sys.exit(f"no cached match pages in {Settings().cache_dir}")
        page = pages[0]
    return page.name, page.read_text(encoding="utf-8")


def main() -> None:
    name, html = load(sys.argv[1:])
    soup = BeautifulSoup(html, "lxml")
    print(f"page: {name} ({len(html):,} chars)")

    games = soup.select(".vm-stats-game")
    print(f"\n.vm-stats-game ids: {[g.get('data-game-id') for g in games]}")
    for g in games[:3]:
        print(f"  game {g.get('data-game-id')}: {len(g.select('table'))} tables inside")
    print(f"tables on page: {[t.get('class') for t in soup.select('table')][:8]}")

    print("\nmarker counts in raw HTML:")
    for marker in MARKERS:
        print(f"  {marker!r}: {html.count(marker)}")

    print("\nlinks/attributes mentioning game= or tab=:")
    seen: set[str] = set()
    for el in soup.find_all(True):
        for attr in ("href", "data-href", "data-url", "data-src"):
            value = el.get(attr)
            if value and re.search(r"[?&](game|tab)=", value) and value not in seen:
                seen.add(value)
                print(f"  <{el.name} {attr}={value!r} class={el.get('class')}>")
    if not seen:
        print("  none")

    print("\nmap-switch nav items:")
    for item in soup.select(".vm-stats-gamesnav-item")[:6]:
        attrs = {k: v for k, v in item.attrs.items() if k != "style"}
        print(f"  {attrs} text={item.get_text(' ', strip=True)!r}")

    scripts = [s.get("src") for s in soup.select("script[src]")]
    print(f"\nscript srcs: {scripts[:10]}")

    game = next((g for g in games if g.get("data-game-id") != "all"), None)
    rows = _player_rows(game) if game is not None else []
    print(f"\nscoreboard rows found in first map: {len(rows)}")
    if rows:
        row = rows[0]
        chain = [f"{a.name}.{'.'.join(a.get('class', []))}" for a in [row, *row.parents][:4]]
        print(f"row element and ancestors: {chain}")
        header = row.find_previous_sibling() or (
            row.parent.find_previous_sibling() if row.parent else None
        )
        if header is not None:
            print("\ncolumn header (element before the first row):")
            print(header.prettify()[:2000])
        print("\nfirst scoreboard row:")
        print(row.prettify()[:5000])

    match = parse_match(html, 0)
    if match.maps and match.maps[0].players:
        print(f"\nfirst parsed player: {match.maps[0].players[0]}")
    print(
        f"\nparser: tags={match.team1.tag!r}/{match.team2.tag!r}, "
        f"players per map={[len(m.players) for m in match.maps]}, "
        f"rounds per map={[len(m.rounds) for m in match.maps]}"
    )


if __name__ == "__main__":
    main()
