"""Turn raw match rows into a clean, model-ready training table."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from wcpredictor.config import Config, default_config


def _outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def build_training_matches(
    matches: pd.DataFrame,
    config: Optional[Config] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Augment raw matches with fields needed by the models.

    Adds:
      * ``result`` in {H, D, A}
      * ``total_goals`` and ``goal_diff`` (home minus away)
      * ``days_ago`` relative to ``reference_date`` (default: latest match)
      * ``recency_weight`` an exponential decay weight based on
        :attr:`PoissonParams.half_life_days`

    The input is not mutated.
    """
    config = config or default_config()
    df = matches.copy()

    df["result"] = [
        _outcome(h, a) for h, a in zip(df["home_score"], df["away_score"])
    ]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"] = df["home_score"] - df["away_score"]

    ref = reference_date if reference_date is not None else df["date"].max()
    df["days_ago"] = (ref - df["date"]).dt.days.clip(lower=0)

    half_life = config.poisson.half_life_days
    if half_life and half_life > 0:
        decay = np.log(2.0) / half_life
        df["recency_weight"] = np.exp(-decay * df["days_ago"])
    else:
        df["recency_weight"] = 1.0

    return df.reset_index(drop=True)


def team_match_counts(matches: pd.DataFrame) -> pd.Series:
    """Number of matches each team has played (home or away)."""
    home = matches["home_team"].value_counts()
    away = matches["away_team"].value_counts()
    return home.add(away, fill_value=0).astype(int).sort_values(ascending=False)
