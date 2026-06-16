"""Generate *synthetic* seed CSVs for the World Cup 2026 predictor.

.. note::
   This is no longer the source of the bundled data -- the repo now ships real
   World Cup results built by ``scripts/fetch_worldcup_data.py``. This script is
   kept as a self-contained, offline fallback that needs no network access.

This produces three files under ``data/raw``:

* ``teams.csv``    - 48 participating teams with confederation + seeding pot.
* ``matches.csv``  - synthetic but realistic international results (2022-2026).
* ``wc2026_groups.csv`` - a plausible group draw (12 groups of 4).

The match results are *synthetic*: they are sampled from a Poisson goals model
driven by per-team latent strengths, so stronger teams genuinely win more
often. This lets the Elo and Poisson models learn meaningful ratings out of the
box while keeping the repo self-contained (no external downloads). Replace
``matches.csv`` with a real dataset to get real-world predictions -- see the
README.

Run with: ``python scripts/generate_seed_data.py``
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# (team, confederation, latent_strength).
# Strength is a unitless skill index calibrated to the FIFA World Rankings as
# of June 11, 2026 (the official pre-tournament ranking). ~1.0 is a top side,
# ~-0.3 is a minnow. Used to generate synthetic historical match results so
# Elo and Poisson learn realistic relative ratings.
TEAMS = [
    # ---- CONCACAF (hosts + qualifiers) ----
    # Hosts carry extra weight; USA/Mexico are also strong by FIFA rank.
    ("USA", "CONCACAF", 0.58),          # FIFA 17
    ("Mexico", "CONCACAF", 0.65),       # FIFA 14
    ("Canada", "CONCACAF", 0.26),       # FIFA 30
    ("Panama", "CONCACAF", 0.18),       # FIFA 34
    ("Curaçao", "CONCACAF", -0.32),     # FIFA 82
    ("Haiti", "CONCACAF", -0.33),       # FIFA 83
    # ---- CONMEBOL ----
    ("Argentina", "CONMEBOL", 1.05),    # FIFA 1
    ("Brazil", "CONMEBOL", 0.88),       # FIFA 6
    ("Colombia", "CONMEBOL", 0.68),     # FIFA ~12
    ("Uruguay", "CONMEBOL", 0.60),      # FIFA 16
    ("Ecuador", "CONMEBOL", 0.40),      # FIFA 23
    ("Paraguay", "CONMEBOL", 0.06),     # FIFA 41
    # ---- OFC ----
    ("New Zealand", "OFC", -0.35),      # FIFA 85
    # ---- AFC ----
    ("Japan", "AFC", 0.55),             # FIFA 18
    ("Iran", "AFC", 0.48),              # FIFA 20
    ("South Korea", "AFC", 0.35),       # FIFA 25
    ("Australia", "AFC", 0.38),         # FIFA 27
    ("Saudi Arabia", "AFC", -0.16),     # FIFA 61
    ("Qatar", "AFC", -0.12),            # FIFA 56
    ("Uzbekistan", "AFC", -0.05),       # FIFA 50
    ("Jordan", "AFC", -0.18),           # FIFA 63
    ("Iraq", "AFC", -0.13),             # FIFA 57
    # ---- CAF ----
    ("Morocco", "CAF", 0.82),           # FIFA 7
    ("Senegal", "CAF", 0.62),           # FIFA 15
    ("Algeria", "CAF", 0.30),           # FIFA 28
    ("Egypt", "CAF", 0.28),             # FIFA 29
    ("Ivory Coast", "CAF", 0.20),       # FIFA 33
    ("Tunisia", "CAF", 0.00),           # FIFA 45
    ("DR Congo", "CAF", -0.02),         # FIFA 46
    ("South Africa", "CAF", -0.15),     # FIFA 60
    ("Cape Verde", "CAF", -0.20),       # FIFA 67
    ("Ghana", "CAF", -0.22),            # FIFA 73
    # ---- UEFA ----
    ("Spain", "UEFA", 1.02),            # FIFA 2
    ("France", "UEFA", 1.00),           # FIFA 3
    ("England", "UEFA", 0.95),          # FIFA 4
    ("Portugal", "UEFA", 0.92),         # FIFA 5
    ("Netherlands", "UEFA", 0.80),      # FIFA 8
    ("Belgium", "UEFA", 0.76),          # FIFA 9
    ("Germany", "UEFA", 0.75),          # FIFA 10
    ("Croatia", "UEFA", 0.70),          # FIFA 11
    ("Switzerland", "UEFA", 0.52),      # FIFA 19
    ("Turkey", "UEFA", 0.45),           # FIFA 22
    ("Austria", "UEFA", 0.38),          # FIFA 24
    ("Norway", "UEFA", 0.24),           # FIFA 31
    ("Scotland", "UEFA", 0.05),         # FIFA 42
    ("Sweden", "UEFA", 0.12),           # FIFA 38
    ("Czech Republic", "UEFA", 0.08),   # FIFA 40
    ("Bosnia & Herzegovina", "UEFA", -0.19),  # FIFA 64
]

# Real 2026 World Cup group draw (Kennedy Center, December 5, 2025).
# Listed in official FIFA seeding order: Pot 1 → Pot 4.
REAL_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia & Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Goals model used to synthesize results.
BASE_LOG_GOALS = 0.10   # exp(0.10) ~ 1.1 baseline goals
HOME_ADVANTAGE = 0.25   # log-scale home boost
STRENGTH_TO_GOALS = 0.55  # how much a strength edge moves expected goals

START_DATE = date(2022, 1, 1)
END_DATE = date(2026, 5, 1)
MATCHES_PER_TEAM = 55


def assign_pots(teams):
    """Sort by strength and split into 4 pots of 12 for the group draw."""
    ranked = sorted(teams, key=lambda t: t[2], reverse=True)
    pots = {}
    for idx, (name, _conf, _s) in enumerate(ranked):
        pots[name] = idx // 12 + 1  # pot 1..4
    return pots


def write_teams(teams, pots):
    path = RAW_DIR / "teams.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["team", "confederation", "pot"])
        for name, conf, _s in sorted(teams, key=lambda t: t[0]):
            writer.writerow([name, conf, pots[name]])
    print(f"wrote {path} ({len(teams)} teams)")


def real_groups():
    """Return the actual 2026 World Cup group draw (no randomness needed)."""
    return {g: list(teams) for g, teams in REAL_GROUPS.items()}


def write_groups(groups):
    path = RAW_DIR / "wc2026_groups.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "slot", "team"])
        for g in sorted(groups):
            for slot, team in enumerate(groups[g], start=1):
                writer.writerow([g, slot, team])
    print(f"wrote {path} ({len(groups)} groups)")


def sample_score(strength_home, strength_away, neutral, rng):
    ha = 0.0 if neutral else HOME_ADVANTAGE
    log_home = BASE_LOG_GOALS + ha + STRENGTH_TO_GOALS * (strength_home - strength_away)
    log_away = BASE_LOG_GOALS - ha + STRENGTH_TO_GOALS * (strength_away - strength_home)
    lam_home = float(np.exp(log_home))
    lam_away = float(np.exp(log_away))
    return int(rng.poisson(lam_home)), int(rng.poisson(lam_away))


def write_matches(teams, rng):
    strength = {name: s for name, _c, s in teams}
    names = [t[0] for t in teams]
    n_matches = len(teams) * MATCHES_PER_TEAM // 2
    total_days = (END_DATE - START_DATE).days

    rows = []
    tournaments = ["Friendly", "Qualifier", "Nations League", "Continental Cup"]
    for _ in range(n_matches):
        i, j = rng.choice(len(names), size=2, replace=False)
        home, away = names[i], names[j]
        tournament = rng.choice(tournaments, p=[0.40, 0.35, 0.15, 0.10])
        # Friendlies are often on neutral ground; qualifiers rarely.
        neutral_prob = 0.35 if tournament == "Friendly" else 0.08
        neutral = bool(rng.random() < neutral_prob)
        hg, ag = sample_score(strength[home], strength[away], neutral, rng)
        offset = int(rng.integers(0, total_days + 1))
        match_date = START_DATE + timedelta(days=offset)
        rows.append((match_date.isoformat(), home, away, hg, ag,
                     int(neutral), tournament))

    rows.sort(key=lambda r: r[0])
    path = RAW_DIR / "matches.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "home_team", "away_team", "home_score",
                         "away_score", "neutral", "tournament"])
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} matches)")


def main():
    # Parse args first so `--help` (or a stray flag) exits cleanly instead of
    # falling through and OVERWRITING data/raw -- exactly the accident this
    # guard prevents.
    argparse.ArgumentParser(
        description=(
            "Regenerate the SYNTHETIC seed CSVs under data/raw "
            "(teams.csv, matches.csv, wc2026_groups.csv). This OVERWRITES "
            "those files. The repo ships REAL World Cup data by default -- "
            "use scripts/fetch_worldcup_data.py to rebuild that instead."
        )
    ).parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    pots = assign_pots(TEAMS)
    write_teams(TEAMS, pots)
    groups = real_groups()
    write_groups(groups)
    write_matches(TEAMS, rng)
    print("done")


if __name__ == "__main__":
    main()
