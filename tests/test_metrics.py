"""Tests for walk-forward backtesting and calibration diagnostics."""

from __future__ import annotations

import warnings

from wcpredictor.data.loader import load_matches
from wcpredictor.data.preprocess import build_training_matches
from wcpredictor.evaluation.metrics import backtest, calibration_table
from wcpredictor.models.elo import EloModel


def test_backtest_collect_predictions_returns_expected_shape(training_matches):
    result = backtest(lambda: EloModel(), training_matches, collect_predictions=True)
    preds = result.predictions
    assert preds is not None
    assert list(preds.columns) == [
        "date",
        "home_team",
        "away_team",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "result",
    ]
    assert len(preds) == result.n_matches


def test_backtest_default_does_not_collect_predictions(training_matches):
    result = backtest(lambda: EloModel(), training_matches)
    assert result.predictions is None


def test_backtest_result_as_dict_excludes_predictions(training_matches):
    result = backtest(lambda: EloModel(), training_matches, collect_predictions=True)
    d = result.as_dict()
    assert "predictions" not in d
    assert set(d) == {"accuracy", "log_loss", "rps", "n_matches"}


def test_calibration_table_bin_counts_and_ranges(training_matches):
    result = backtest(lambda: EloModel(), training_matches, collect_predictions=True)
    calib = calibration_table(result.predictions)

    assert calib["count"].sum() == 3 * len(result.predictions)
    assert (calib["count"] > 0).all()
    assert calib["mean_predicted"].between(0.0, 1.0).all()
    assert calib["empirical_freq"].between(0.0, 1.0).all()
    assert set(calib["outcome"]) <= {"H", "D", "A"}


def test_backtest_no_longer_prints_unseen_team_warnings():
    """Walk-forward refits meet historical debutants at early cutoffs on the
    real data; those "unseen team" UserWarnings are expected but must not
    reach the terminal by default (they used to spam ~80 lines)."""
    matches = load_matches()
    training = build_training_matches(matches)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # A small min_train deliberately walks through many historical
        # debutants, so this data/setting combination is known to trigger
        # the warning if it isn't suppressed.
        backtest(lambda: EloModel(), training, min_train=50, refit_every=50)

    unseen_warnings = [
        w for w in caught if "not seen in training data" in str(w.message)
    ]
    assert unseen_warnings == []
