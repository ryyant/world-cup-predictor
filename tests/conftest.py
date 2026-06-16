"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredictor.data.preprocess import build_training_matches


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
