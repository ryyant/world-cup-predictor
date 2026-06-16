"""Build the bundled seed CSVs from real World Cup data (openfootball/worldcup.json).

Downloads each tournament's ``worldcup.json`` (vendored under ``data/source``),
parses the finals results, and regenerates the three files under ``data/raw``:

* ``matches.csv``       - real World Cup finals results, 1930-2026 (including any
                          already-played 2026 group games). Unplayed/knockout
                          fixtures whose teams are still placeholders (e.g.
                          ``"W89"``, ``"2A"``) or that have no full-time score are
                          skipped.
* ``wc2026_groups.csv`` - the real 2026 group draw (12 groups of 4), taken from
                          ``2026/worldcup.json``.
* ``teams.csv``         - the 48 qualified teams with confederation and a derived
                          strength-tier ``pot`` (descriptive metadata only; the
                          simulator reads the drawn groups, not the pots).

Because the source covers only World Cup *finals* (no qualifiers or friendlies),
a handful of 2026 debutants (Cape Verde, Curacao, Jordan, Uzbekistan) have no
prior matches and fall back to the models' default ratings.

Run with::

    python scripts/fetch_worldcup_data.py [--refresh]

``--refresh`` re-downloads every year file even if it is already vendored.

Source: https://github.com/openfootball/worldcup.json (data released under CC0).
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SOURCE_DIR = PROJECT_ROOT / "data" / "source"

BASE_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"

# National World Cup years (no tournament in 1942/1946). 2026 additionally
# supplies the group draw and any already-played group games.
YEARS = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978,
         1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]

# Canonicalise historical names to the spelling used in the 2026 draw so a
# team's record links up across eras and source inconsistencies.
ALIASES = {
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
    "United States": "USA",
    "West Germany": "Germany",
    "Zaire": "DR Congo",
}

# Host nation(s) per tournament (post-alias spellings). A World Cup match is
# treated as non-neutral only when a host is playing.
HOSTS = {
    1930: {"Uruguay"}, 1934: {"Italy"}, 1938: {"France"}, 1950: {"Brazil"},
    1954: {"Switzerland"}, 1958: {"Sweden"}, 1962: {"Chile"}, 1966: {"England"},
    1970: {"Mexico"}, 1974: {"Germany"}, 1978: {"Argentina"}, 1982: {"Spain"},
    1986: {"Mexico"}, 1990: {"Italy"}, 1994: {"USA"}, 1998: {"France"},
    2002: {"South Korea", "Japan"}, 2006: {"Germany"}, 2010: {"South Africa"},
    2014: {"Brazil"}, 2018: {"Russia"}, 2022: {"Qatar"},
    2026: {"USA", "Mexico", "Canada"},
}

# Confederation of each 2026 qualifier (factual). build_teams() fails loudly if
# the drawn teams don't all appear here, so a name change shows up immediately.
CONFEDERATION = {
    # CONCACAF
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Haiti": "CONCACAF", "Curaçao": "CONCACAF", "Panama": "CONCACAF",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    # UEFA
    "Germany": "UEFA", "Spain": "UEFA", "France": "UEFA", "England": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA",
    "Scotland": "UEFA", "Norway": "UEFA", "Sweden": "UEFA", "Turkey": "UEFA",
    "Czech Republic": "UEFA", "Bosnia & Herzegovina": "UEFA",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Tunisia": "CAF", "Egypt": "CAF",
    "Algeria": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF",
    "South Africa": "CAF", "Cape Verde": "CAF", "DR Congo": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Uzbekistan": "AFC",
    "Jordan": "AFC",
    # OFC
    "New Zealand": "OFC",
}


def canon(team: str | None) -> str | None:
    """Return the canonical (2026-draw) spelling of a team name."""
    return ALIASES.get(team, team)


def is_placeholder(team: str | None) -> bool:
    """openfootball fills unplayed knockout slots with non-country tokens such as
    ``"W89"``, ``"L73"`` or ``"2A"``. Real country names never contain a digit."""
    return team is None or any(ch.isdigit() for ch in team)


def download(year: int, refresh: bool = False) -> Path:
    """Return the local path to ``year``'s vendored JSON, downloading if absent."""
    path = SOURCE_DIR / f"{year}.worldcup.json"
    if refresh or not path.exists():
        url = BASE_URL.format(year=year)
        req = urllib.request.Request(url, headers={"User-Agent": "wcpredictor-fetch"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - fixed trusted host
            path.write_bytes(resp.read())
        print(f"downloaded {url}")
    return path


def iter_matches(doc: dict):
    """Yield match dicts from either a flat ``matches`` list or ``rounds[*].matches``."""
    if isinstance(doc.get("matches"), list):
        yield from doc["matches"]
    for rnd in doc.get("rounds", []):
        yield from rnd.get("matches", [])


def parse_played(match: dict):
    """Return ``(home, away, home_goals, away_goals)`` for a completed, real-team
    match, or ``None`` to skip it (placeholder team or missing full-time score)."""
    t1, t2 = match.get("team1"), match.get("team2")
    if is_placeholder(t1) or is_placeholder(t2):
        return None
    ft = (match.get("score") or {}).get("ft")
    if not (isinstance(ft, list) and len(ft) == 2):
        return None
    try:
        hg, ag = int(ft[0]), int(ft[1])
    except (TypeError, ValueError):
        return None
    if hg < 0 or ag < 0:
        return None
    return canon(t1), canon(t2), hg, ag


def build_matches(docs: dict) -> list:
    """Write data/raw/matches.csv and return the rows (for strength scoring)."""
    rows = []
    for year, doc in docs.items():
        hosts = HOSTS.get(year, set())
        for m in iter_matches(doc):
            parsed = parse_played(m)
            date = m.get("date")
            if parsed is None or not date:
                continue
            home, away, hg, ag = parsed
            neutral = 0 if (home in hosts or away in hosts) else 1
            rows.append((date, home, away, hg, ag, neutral, f"World Cup {year}"))
    rows.sort(key=lambda r: r[0])
    path = RAW_DIR / "matches.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "home_team", "away_team", "home_score",
                    "away_score", "neutral", "tournament"])
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} matches)")
    return rows


def build_groups(doc2026: dict) -> list:
    """Write data/raw/wc2026_groups.csv from the 2026 fixtures; return the teams."""
    groups: dict[str, list[str]] = {}
    for m in iter_matches(doc2026):
        label = m.get("group")
        if not label:
            continue
        letter = label.replace("Group", "").strip()
        for t in (m.get("team1"), m.get("team2")):
            if is_placeholder(t):
                continue
            team = canon(t)
            bucket = groups.setdefault(letter, [])
            if team not in bucket:
                bucket.append(team)
    if len(groups) != 12 or any(len(v) != 4 for v in groups.values()):
        sizes = {g: len(v) for g, v in sorted(groups.items())}
        raise SystemExit(f"expected 12 groups of 4, got {sizes}")
    path = RAW_DIR / "wc2026_groups.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "slot", "team"])
        for g in sorted(groups):
            for slot, team in enumerate(sorted(groups[g]), start=1):
                w.writerow([g, slot, team])
    teams = sorted({t for v in groups.values() for t in v})
    print(f"wrote {path} ({len(groups)} groups, {len(teams)} teams)")
    return teams


def derive_pots(teams: list, rows: list) -> dict:
    """Rank teams by average goal difference per finals match (0.0 with no
    history) and split into four tiers of 12. Descriptive metadata only."""
    diff: dict[str, int] = defaultdict(int)
    games: dict[str, int] = defaultdict(int)
    for _date, home, away, hg, ag, _neutral, _tour in rows:
        diff[home] += hg - ag
        games[home] += 1
        diff[away] += ag - hg
        games[away] += 1
    score = {t: (diff[t] / games[t] if games[t] else 0.0) for t in teams}
    ranked = sorted(teams, key=lambda t: score[t], reverse=True)
    return {t: i // 12 + 1 for i, t in enumerate(ranked)}


def build_teams(teams: list, rows: list) -> None:
    """Write data/raw/teams.csv (team, confederation, pot)."""
    missing = [t for t in teams if t not in CONFEDERATION]
    if missing:
        raise SystemExit(f"no confederation mapping for: {missing} "
                         "- add them to CONFEDERATION (or ALIASES).")
    pots = derive_pots(teams, rows)
    path = RAW_DIR / "teams.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team", "confederation", "pot"])
        for t in sorted(teams):
            w.writerow([t, CONFEDERATION[t], pots[t]])
    print(f"wrote {path} ({len(teams)} teams)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build seed CSVs from openfootball/worldcup.json")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download every year file even if already vendored")
    args = ap.parse_args()

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    docs = {}
    for year in YEARS:
        path = download(year, refresh=args.refresh)
        docs[year] = json.loads(path.read_text(encoding="utf-8"))

    rows = build_matches(docs)
    teams = build_groups(docs[2026])
    build_teams(teams, rows)
    print("done")


if __name__ == "__main__":
    main()
