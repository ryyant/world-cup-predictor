"""Tests for the Elo model."""

from __future__ import annotations

import pytest

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
