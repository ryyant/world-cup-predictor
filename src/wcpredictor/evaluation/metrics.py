"""Backtesting metrics for match-outcome models.

Walks forward through historical matches: for each match the model is asked for
win/draw/loss probabilities *before* the result is known, then we score those
probabilities against the actual outcome using accuracy, multiclass log-loss and
the ranked probability score (RPS, the standard metric for ordered football
outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from wcpredictor.config import Config, default_config
from wcpredictor.models.base import MatchModel

OUTCOMES = ("H", "D", "A")


@dataclass
class BacktestResult:
    accuracy: float
    log_loss: float
    rps: float
    n_matches: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "rps": self.rps,
            "n_matches": self.n_matches,
        }

    def __str__(self) -> str:
        return (
            f"matches={self.n_matches}  accuracy={self.accuracy:.3f}  "
            f"log_loss={self.log_loss:.3f}  rps={self.rps:.3f}"
        )


def _ranked_probability_score(probs: np.ndarray, actual_idx: int) -> float:
    """RPS for a single ordered 3-outcome prediction (H, D, A)."""
    outcome = np.zeros(3)
    outcome[actual_idx] = 1.0
    cum_p = np.cumsum(probs)
    cum_o = np.cumsum(outcome)
    # Divide by (categories - 1) so RPS is in [0, 1].
    return float(np.sum((cum_p - cum_o) ** 2) / (len(probs) - 1))


def backtest(
    model_factory,
    training_matches: pd.DataFrame,
    config: Optional[Config] = None,
    min_train: int = 200,
    refit_every: int = 100,
) -> BacktestResult:
    """Walk-forward backtest.

    ``model_factory`` is a zero-arg callable returning a fresh, unfitted model
    (e.g. ``lambda: EloModel()``). The model is fit on all matches before the
    current evaluation block and refit every ``refit_every`` matches for speed.
    """
    config = config or default_config()
    df = training_matches.sort_values("date").reset_index(drop=True)
    eps = 1e-15

    accs: List[float] = []
    losses: List[float] = []
    rps_scores: List[float] = []

    model: Optional[MatchModel] = None
    n = len(df)
    i = min_train
    while i < n:
        # Refit on everything seen so far at each block boundary.
        model = model_factory().fit(df.iloc[:i])
        block_end = min(i + refit_every, n)
        for j in range(i, block_end):
            row = df.iloc[j]
            pred = model.predict_match(
                row["home_team"], row["away_team"], bool(row["neutral"])
            )
            probs = np.array(
                [pred.p_home_win, pred.p_draw, pred.p_away_win]
            )
            actual_idx = OUTCOMES.index(row["result"])
            predicted_idx = int(np.argmax(probs))
            accs.append(1.0 if predicted_idx == actual_idx else 0.0)
            losses.append(-np.log(max(probs[actual_idx], eps)))
            rps_scores.append(_ranked_probability_score(probs, actual_idx))
        i = block_end

    if not accs:
        raise ValueError(
            "Not enough matches to backtest; reduce min_train or add data."
        )

    return BacktestResult(
        accuracy=float(np.mean(accs)),
        log_loss=float(np.mean(losses)),
        rps=float(np.mean(rps_scores)),
        n_matches=len(accs),
    )
