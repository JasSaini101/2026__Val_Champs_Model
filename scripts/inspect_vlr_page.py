"""Show how a cached vlr.gg match page is structured, to debug parser selectors.

Usage:
    uv run python scripts/inspect_vlr_page.py                # first cached match page
    uv run python scripts/inspect_vlr_page.py path/to/page.html
"""

from __future__ import annotations

import sys
from pathlib import Path

from bs4 import BeautifulSoup

from valchamps.config import Settings
from valchamps.data.parser import parse_match


def pick_page(args: list[str]) -> Path:
    if args:
        return Path(args[0])
    pages = sorted(p for p in Settings().cache_dir.glob("*.html") if p.name[0].isdigit())
    if not pages:
        sys.exit(f"no cached match pages in {Settings().cache_dir}")
    return pages[0]


def main() -> None:
    page = pick_page(sys.argv[1:])
    html = page.read_text(encoding="utf-8")
    print(f"file: {page.name} ({len(html):,} chars)")

    for parser in ("lxml", "html.parser"):
        soup = BeautifulSoup(html, parser)
        games = soup.select(".vm-stats-game")
        print(f"\n[{parser}] .vm-stats-game ids: {[g.get('data-game-id') for g in games]}")
        for g in games[:2]:
            classes = [t.get("class") for t in g.select("table")]
            print(f"  game {g.get('data-game-id')}: tables inside = {classes}")
        print(f"  all tables on page: {[t.get('class') for t in soup.select('table')][:8]}")

    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table")
    row = table.select_one("tr:has(td)") if table else None
    print("\nfirst data row of first table:")
    print(row.prettify()[:3000] if row else "  <no table rows found>")

    game = next((g for g in soup.select(".vm-stats-game") if g.get("data-game-id") != "all"), None)
    if game is not None and not game.select("table"):
        print("\nfirst map block has no tables; its markup after the header:")
        header = game.select_one(".vm-stats-game-header")
        rest = header.find_next_sibling() if header else None
        print(rest.prettify()[:3000] if rest else "  <nothing after header>")

    match = parse_match(html, 0)
    print(
        f"\nparser: tags={match.team1.tag!r}/{match.team2.tag!r}, "
        f"players per map={[len(m.players) for m in match.maps]}"
    )


if __name__ == "__main__":
    main()
