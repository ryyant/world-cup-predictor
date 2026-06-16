"""Tests for the Poisson model and single-match simulation."""

from __future__ import annotations

import numpy as np
import pytest

from wcpredictor.models.poisson import PoissonModel
from wcpredictor.simulation.match import simulate_match


def test_predict_raises_before_fit():
    model = PoissonModel()
    with pytest.raises(RuntimeError):
        model.expected_goals("Strong", "Weak")


def test_strengths_reflect_quality(training_matches):
    model = PoissonModel().fit(training_matches)
    strong = model.attack["Strong"] + model.defense["Strong"]
    weak = model.attack["Weak"] + model.defense["Weak"]
    assert strong > weak


def test_expected_goals_favor_stronger(training_matches):
    model = PoissonModel().fit(training_matches)
    lam_strong, lam_weak = model.expected_goals("Strong", "Weak", neutral=True)
    assert lam_strong > lam_weak


def test_scoreline_matrix_is_distribution(training_matches):
    model = PoissonModel().fit(training_matches)
    matrix = model.scoreline_matrix("Strong", "Weak", neutral=True)
    assert matrix.min() >= 0
    assert pytest.approx(matrix.sum(), abs=1e-9) == 1.0


def test_predict_probabilities_sum_to_one(training_matches):
    model = PoissonModel().fit(training_matches)
    pred = model.predict_match("Strong", "Weak", neutral=True)
    total = pred.p_home_win + pred.p_draw + pred.p_away_win
    assert pytest.approx(total, abs=1e-9) == 1.0
    assert pred.p_home_win > pred.p_away_win


def test_save_load_roundtrip(tmp_path, training_matches):
    model = PoissonModel().fit(training_matches)
    path = tmp_path / "poisson.csv"
    model.save(path)
    loaded = PoissonModel.load(path)
    assert loaded.intercept == pytest.approx(model.intercept)
    assert loaded.home_advantage == pytest.approx(model.home_advantage)
    a, b = model.expected_goals("Strong", "Weak")
    a2, b2 = loaded.expected_goals("Strong", "Weak")
    assert a == pytest.approx(a2) and b == pytest.approx(b2)


def test_group_match_can_draw(training_matches):
    model = PoissonModel().fit(training_matches)
    rng = np.random.default_rng(1)
    saw_draw = any(
        simulate_match(model, "Medium", "Medium", rng, knockout=False).is_draw
        for _ in range(200)
    )
    assert saw_draw


def test_knockout_never_draws(training_matches):
    model = PoissonModel().fit(training_matches)
    rng = np.random.default_rng(2)
    for _ in range(200):
        res = simulate_match(model, "Medium", "Medium", rng, knockout=True)
        assert res.winner in ("Medium",)
        assert not res.is_draw
