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

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# (team, confederation, latent_strength). Strength is a unitless skill index;
# ~1.0 is a top side, ~-1.0 is a minnow. Used only to generate results.
TEAMS = [
    # CONCACAF (hosts + qualifiers)
    ("USA", "CONCACAF", 0.25),
    ("Mexico", "CONCACAF", 0.30),
    ("Canada", "CONCACAF", 0.10),
    ("Costa Rica", "CONCACAF", -0.30),
    ("Panama", "CONCACAF", -0.40),
    ("Jamaica", "CONCACAF", -0.45),
    # UEFA
    ("France", "UEFA", 1.05),
    ("England", "UEFA", 0.95),
    ("Spain", "UEFA", 0.98),
    ("Germany", "UEFA", 0.85),
    ("Portugal", "UEFA", 0.90),
    ("Netherlands", "UEFA", 0.82),
    ("Belgium", "UEFA", 0.72),
    ("Italy", "UEFA", 0.78),
    ("Croatia", "UEFA", 0.60),
    ("Denmark", "UEFA", 0.55),
    ("Switzerland", "UEFA", 0.48),
    ("Austria", "UEFA", 0.45),
    ("Ukraine", "UEFA", 0.40),
    ("Poland", "UEFA", 0.38),
    ("Serbia", "UEFA", 0.42),
    ("Norway", "UEFA", 0.50),
    ("Scotland", "UEFA", 0.30),
    ("Hungary", "UEFA", 0.35),
    # CONMEBOL
    ("Brazil", "CONMEBOL", 1.00),
    ("Argentina", "CONMEBOL", 1.08),
    ("Uruguay", "CONMEBOL", 0.68),
    ("Colombia", "CONMEBOL", 0.62),
    ("Ecuador", "CONMEBOL", 0.40),
    ("Paraguay", "CONMEBOL", 0.20),
    # AFC
    ("Japan", "AFC", 0.55),
    ("South Korea", "AFC", 0.50),
    ("Iran", "AFC", 0.42),
    ("Australia", "AFC", 0.40),
    ("Saudi Arabia", "AFC", 0.18),
    ("Qatar", "AFC", 0.10),
    ("Iraq", "AFC", 0.00),
    ("Uzbekistan", "AFC", 0.05),
    # CAF
    ("Morocco", "CAF", 0.70),
    ("Senegal", "CAF", 0.58),
    ("Nigeria", "CAF", 0.52),
    ("Egypt", "CAF", 0.45),
    ("Cameroon", "CAF", 0.35),
    ("Ghana", "CAF", 0.33),
    ("Algeria", "CAF", 0.48),
    ("Ivory Coast", "CAF", 0.46),
    ("Tunisia", "CAF", 0.30),
    # OFC
    ("New Zealand", "OFC", -0.20),
]

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


def draw_groups(teams, pots, rng):
    """Build 12 groups of 4 with one team per pot, respecting a simple
    confederation constraint (max one team per confederation per group, except
    UEFA which may have up to two)."""
    by_pot = {1: [], 2: [], 3: [], 4: []}
    for name, conf, _s in teams:
        by_pot[pots[name]].append((name, conf))
    for pot in by_pot:
        rng.shuffle(by_pot[pot])

    group_letters = [chr(ord("A") + i) for i in range(12)]
    groups = {g: [] for g in group_letters}
    group_confs = {g: [] for g in group_letters}

    def can_place(group, conf):
        existing = group_confs[group]
        if conf == "UEFA":
            return existing.count("UEFA") < 2
        return conf not in existing

    for pot in (1, 2, 3, 4):
        pool = by_pot[pot]
        # Greedy assignment with backtracking-lite: try to place each team in
        # the first group (that still needs a pot-`pot` team) it is allowed in.
        for name, conf in pool:
            placed = False
            order = group_letters[:]
            rng.shuffle(order)
            for g in order:
                if len(groups[g]) == pot - 1 and can_place(g, conf):
                    groups[g].append(name)
                    group_confs[g].append(conf)
                    placed = True
                    break
            if not placed:
                # Fallback: drop the confederation constraint to guarantee a
                # complete draw.
                for g in order:
                    if len(groups[g]) == pot - 1:
                        groups[g].append(name)
                        group_confs[g].append(conf)
                        placed = True
                        break
    return groups


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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    pots = assign_pots(TEAMS)
    write_teams(TEAMS, pots)
    groups = draw_groups(TEAMS, pots, rng)
    write_groups(groups)
    write_matches(TEAMS, rng)
    print("done")


if __name__ == "__main__":
    main()
