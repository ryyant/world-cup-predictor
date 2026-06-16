"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredictor.data.preprocess import build_training_matches
from wcpredictor.models.poisson import PoissonModel


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    """A small synthetic results table with a clear strength hierarchy.

    Strong > Medium > Weak. Generated deterministically so tests are stable.
    """
    rng = np.random.default_rng(0)
    strength = {"Strong": 1.0, "Medium": 0.0, "Weak": -1.0}
    teams = list(strength)
    rows = []
    start = pd.Timestamp("2023-01-01")
    for i in range(600):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = np.exp(0.1 + 0.5 * (strength[h] - strength[a]))
        lam_a = np.exp(0.1 + 0.5 * (strength[a] - strength[h]))
        rows.append(
            {
                "date": start + pd.Timedelta(days=i),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(lam_h)),
                "away_score": int(rng.poisson(lam_a)),
                "neutral": True,
                "tournament": "Friendly",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def training_matches(synthetic_matches) -> pd.DataFrame:
    return build_training_matches(synthetic_matches)


@pytest.fixture(scope="session")
def groups48() -> pd.DataFrame:
    """48 teams across 12 groups; team names encode a strength tier."""
    rows = []
    letters = [chr(ord("A") + i) for i in range(12)]
    n = 0
    for g in letters:
        for slot in range(1, 5):
            rows.append({"group": g, "slot": slot, "team": f"T{n:02d}"})
            n += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiered_model(groups48) -> PoissonModel:
    """A Poisson model whose strengths increase with team index."""
    model = PoissonModel()
    teams = list(groups48["team"])
    model.attack = {t: 0.02 * i for i, t in enumerate(teams)}
    model.defense = {t: 0.02 * i for i, t in enumerate(teams)}
    model.intercept = np.log(1.3)
    model.home_advantage = np.log(1.1)
    return model
