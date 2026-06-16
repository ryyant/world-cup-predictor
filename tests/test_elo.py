"""Tests for the Elo model."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from wcpredictor.data.loader import load_matches
from wcpredictor.data.preprocess import build_training_matches
from wcpredictor.models.elo import EloModel


def test_predict_raises_before_fit():
    model = EloModel()
    with pytest.raises(RuntimeError):
        model.predict_match("Strong", "Weak")


def test_ratings_reflect_strength(training_matches):
    model = EloModel().fit(training_matches)
    assert model.rating("Strong") > model.rating("Medium") > model.rating("Weak")


def test_probabilities_are_valid_distribution(training_matches):
    model = EloModel().fit(training_matches)
    pred = model.predict_match("Strong", "Weak", neutral=True)
    probs = [pred.p_home_win, pred.p_draw, pred.p_away_win]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert pytest.approx(sum(probs), abs=1e-9) == 1.0


def test_stronger_team_more_likely_to_win(training_matches):
    model = EloModel().fit(training_matches)
    pred = model.predict_match("Strong", "Weak", neutral=True)
    assert pred.p_home_win > pred.p_away_win
    assert pred.most_likely == "H"


def test_home_advantage_increases_win_prob(training_matches):
    model = EloModel().fit(training_matches)
    home = model.predict_match("Medium", "Medium", neutral=False)
    neutral = model.predict_match("Medium", "Medium", neutral=True)
    assert home.p_home_win > neutral.p_home_win


def test_even_match_is_symmetric_on_neutral(training_matches):
    model = EloModel().fit(training_matches)
    pred = model.predict_match("Medium", "Medium", neutral=True)
    assert pred.p_home_win == pytest.approx(pred.p_away_win, abs=1e-9)


def test_save_and_load_roundtrip(tmp_path, training_matches):
    model = EloModel().fit(training_matches)
    path = tmp_path / "elo.csv"
    model.save(path)
    loaded = EloModel.load(path)
    assert loaded.rating("Strong") == pytest.approx(model.rating("Strong"))


def test_unseen_team_warns_once(training_matches):
    model = EloModel().fit(training_matches)
    with pytest.warns(UserWarning, match="not seen in training data"):
        model.rating("Atlantis")
    # Second lookup for the same team must be silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.rating("Atlantis")


def test_save_load_preserves_fitted_draw_base(tmp_path, training_matches):
    """Round-trips a genuinely fitted value (not a manually-assigned one)."""
    model = EloModel().fit(training_matches)
    assert model.fitted_draw_base is not None
    path = tmp_path / "elo.csv"
    model.save(path)
    loaded = EloModel.load(path)
    assert loaded.fitted_draw_base == pytest.approx(model.fitted_draw_base)
    assert loaded.rating("Strong") == pytest.approx(model.rating("Strong"))


def test_load_tolerates_legacy_csv_without_meta(tmp_path):
    path = tmp_path / "legacy_elo.csv"
    pd.DataFrame(
        {"team": ["Strong", "Weak"], "rating": [1600.0, 1400.0]}
    ).to_csv(path, index=False)
    loaded = EloModel.load(path)
    assert loaded.fitted_draw_base is None
    assert loaded.rating("Strong") == pytest.approx(1600.0)
    assert set(loaded.ratings) == {"Strong", "Weak"}


def test_ratings_table_excludes_meta_rows(tmp_path, training_matches):
    model = EloModel().fit(training_matches)
    path = tmp_path / "elo.csv"
    model.save(path)
    table = EloModel.load(path).ratings_table()
    assert not table["team"].astype(str).str.startswith("__").any()


def test_track_history_records_every_rating_update(training_matches):
    model = EloModel().fit(training_matches, track_history=True)
    assert model.history is not None
    assert list(model.history.columns) == ["date", "team", "rating"]
    assert len(model.history) == 2 * len(training_matches)


def test_track_history_defaults_to_none(training_matches):
    model = EloModel().fit(training_matches)
    assert model.history is None


def test_track_history_does_not_change_fitted_ratings(training_matches):
    """track_history is pure bookkeeping; it must not alter model output."""
    plain = EloModel().fit(training_matches)
    tracked = EloModel().fit(training_matches, track_history=True)
    assert tracked.ratings == pytest.approx(plain.ratings)
    assert tracked.fitted_draw_base == pytest.approx(plain.fitted_draw_base)


def test_fitted_draw_base_satisfies_moment_condition():
    """Mean predicted p_draw over the training matches should track the
    empirical draw rate (the moment condition fitted_draw_base is fit to
    satisfy), on the real 1930-2026 finals data where the clip does not
    bind."""
    matches = load_matches()
    training = build_training_matches(matches)
    model = EloModel().fit(training)

    lo, hi = model.params.fitted_draw_base_bounds
    assert lo < model.fitted_draw_base < hi  # clip is not binding here

    predicted = [
        model.predict_match(r.home_team, r.away_team, bool(r.neutral)).p_draw
        for r in training.itertuples(index=False)
    ]
    empirical_draw_rate = (training["home_score"] == training["away_score"]).mean()
    # The moment condition is fit exactly using PRE-match (walk-forward)
    # ratings; this check re-evaluates p_draw with the FINAL ratings, so it is
    # only an approximation of that condition. The gap widens a little as more
    # decisive recent results (e.g. the 2026 knockout rounds) sharpen the final
    # ratings, hence the 0.03 tolerance.
    assert np.mean(predicted) == pytest.approx(empirical_draw_rate, abs=0.03)
