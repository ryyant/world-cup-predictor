"""Backtesting metrics for match-outcome models.

Walks forward through historical matches: for each match the model is asked for
win/draw/loss probabilities *before* the result is known, then we score those
probabilities against the actual outcome using accuracy, multiclass log-loss and
the ranked probability score (RPS, the standard metric for ordered football
outcomes).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from wcpredictor.config import Config, default_config
from wcpredictor.data.preprocess import recency_weights
from wcpredictor.models.base import MatchModel

OUTCOMES = ("H", "D", "A")

# Default number of matches used to warm up the model before scoring begins.
DEFAULT_MIN_TRAIN = 300

# predicted_prob column per outcome, used by calibration_table to reshape
# backtest predictions from wide to long.
_OUTCOME_PROB_COLUMNS = {"H": "p_home_win", "D": "p_draw", "A": "p_away_win"}


@dataclass
class BacktestResult:
    accuracy: float
    log_loss: float
    rps: float
    n_matches: int
    # Per-match predictions, only populated when backtest(collect_predictions=True)
    # is used. A diagnostic artifact (feeds calibration_table), not a scalar
    # metric, so it's left out of as_dict()/__str__().
    predictions: Optional[pd.DataFrame] = None

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
    min_train: int = DEFAULT_MIN_TRAIN,
    refit_every: int = 100,
    collect_predictions: bool = False,
) -> BacktestResult:
    """Walk-forward backtest.

    ``model_factory`` is a zero-arg callable returning a fresh, unfitted model
    (e.g. ``lambda: EloModel()``). The model is fit on all matches before the
    current evaluation block and refit every ``refit_every`` matches for speed.

    Set ``collect_predictions=True`` to additionally record each scored
    match's date, teams, predicted probabilities and actual result on
    ``BacktestResult.predictions`` (e.g. as input to :func:`calibration_table`).
    """
    config = config or default_config()
    df = training_matches.sort_values("date").reset_index(drop=True)
    eps = 1e-15

    accs: List[float] = []
    losses: List[float] = []
    rps_scores: List[float] = []
    pred_rows: List[dict] = []

    model: Optional[MatchModel] = None
    n = len(df)
    i = min_train
    while i < n:
        # Walk-forward refits meet historical debutants at early cutoffs --
        # each model's "unseen team" UserWarning is expected here, not a bug,
        # so it's suppressed tightly around just the fit+predict calls below
        # (not package-wide, and not anywhere else in the codebase).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Refit on everything seen so far at each block boundary. Recency
            # weights are recomputed relative to this cutoff (the last match
            # the model has "seen") rather than reused from the full-history
            # column, whose future reference date would uniformly shrink an
            # early slice and let the ridge penalty over-shrink the fit.
            train_slice = df.iloc[:i].copy()
            train_slice["recency_weight"] = recency_weights(
                train_slice["date"], config,
                reference_date=train_slice["date"].max(),
            )
            model = model_factory().fit(train_slice)
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
                if collect_predictions:
                    pred_rows.append(
                        {
                            "date": row["date"],
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "p_home_win": pred.p_home_win,
                            "p_draw": pred.p_draw,
                            "p_away_win": pred.p_away_win,
                            "result": row["result"],
                        }
                    )
        i = block_end

    if not accs:
        raise ValueError(
            "Not enough matches to backtest; reduce min_train or add data."
        )

    predictions = pd.DataFrame(pred_rows) if collect_predictions else None
    return BacktestResult(
        accuracy=float(np.mean(accs)),
        log_loss=float(np.mean(losses)),
        rps=float(np.mean(rps_scores)),
        n_matches=len(accs),
        predictions=predictions,
    )


def calibration_table(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bin predicted probabilities against realized outcome frequencies.

    Reshapes ``predictions`` (one row per match, with wide ``p_home_win`` /
    ``p_draw`` / ``p_away_win`` / ``result`` columns, as produced by
    ``backtest(collect_predictions=True)``) into one row per (match, outcome)
    with a ``predicted_prob`` and a ``did_occur`` indicator, then bins
    ``predicted_prob`` into ``n_bins`` equal-width bins over [0, 1] separately
    for each outcome. A well-calibrated model has ``mean_predicted`` roughly
    equal to ``empirical_freq`` in every bin -- the data behind a reliability
    diagram.
    """
    long_frames = []
    for outcome in OUTCOMES:
        predicted_prob = predictions[_OUTCOME_PROB_COLUMNS[outcome]]
        did_occur = (predictions["result"] == outcome).astype(float)
        long_frames.append(
            pd.DataFrame(
                {
                    "outcome": outcome,
                    "predicted_prob": predicted_prob.to_numpy(),
                    "did_occur": did_occur.to_numpy(),
                }
            )
        )
    long = pd.concat(long_frames, ignore_index=True)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    long["bin"] = pd.cut(long["predicted_prob"], bins=bin_edges, include_lowest=True)

    # observed=True drops (outcome, bin) combinations with no rows, i.e. empty
    # bins, rather than materializing them with count 0.
    table = (
        long.groupby(["outcome", "bin"], observed=True)
        .agg(
            mean_predicted=("predicted_prob", "mean"),
            empirical_freq=("did_occur", "mean"),
            count=("did_occur", "size"),
        )
        .reset_index()
    )
    table["bin_mid"] = table["bin"].apply(lambda interval: interval.mid).astype(float)
    return (
        table[["outcome", "bin_mid", "mean_predicted", "empirical_freq", "count"]]
        .sort_values(["outcome", "bin_mid"])
        .reset_index(drop=True)
    )
