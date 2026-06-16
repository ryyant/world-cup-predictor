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


def recency_weights(
    dates: pd.Series,
    config: Optional[Config] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """Exponential-decay recency weights for a series of match dates.

    A match played ``half_life_days`` before ``reference_date`` (default: the
    most recent date in ``dates``) is weighted half as much as one played on
    the reference date itself. ``half_life_days <= 0`` disables decay (all
    weights 1).

    The scale is anchored to ``reference_date``, so callers that refit at a
    historical cutoff (notably the walk-forward :func:`backtest`) must pass
    that cutoff as ``reference_date``. Otherwise every weight is measured
    against a fixed future date and the whole slice is scaled down uniformly;
    because the Poisson likelihood is weighted but the ridge penalty is not,
    that uniform shrink makes the penalty dominate and over-shrinks the fit.
    """
    config = config or default_config()
    ref = reference_date if reference_date is not None else dates.max()
    days_ago = (ref - dates).dt.days.clip(lower=0)
    half_life = config.poisson.half_life_days
    if half_life and half_life > 0:
        decay = np.log(2.0) / half_life
        return np.exp(-decay * days_ago).to_numpy(dtype=float)
    return np.ones(len(dates), dtype=float)


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
    df["recency_weight"] = recency_weights(df["date"], config, reference_date=ref)

    return df.reset_index(drop=True)


def team_match_counts(matches: pd.DataFrame) -> pd.Series:
    """Number of matches each team has played (home or away)."""
    home = matches["home_team"].value_counts()
    away = matches["away_team"].value_counts()
    return home.add(away, fill_value=0).astype(int).sort_values(ascending=False)
