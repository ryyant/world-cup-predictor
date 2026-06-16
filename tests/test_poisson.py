"""Tests for the Poisson model and single-match simulation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson as poisson_dist

from wcpredictor.config import Config, PoissonParams
from wcpredictor.data.loader import load_matches
from wcpredictor.data.preprocess import build_training_matches
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
    assert loaded.rho == pytest.approx(model.rho)
    a, b = model.expected_goals("Strong", "Weak")
    a2, b2 = loaded.expected_goals("Strong", "Weak")
    assert a == pytest.approx(a2) and b == pytest.approx(b2)


def test_load_tolerates_legacy_csv_without_rho(tmp_path, training_matches):
    """CSVs saved before the Dixon-Coles rho meta row existed still load."""
    model = PoissonModel().fit(training_matches)
    path = tmp_path / "poisson.csv"
    model.save(path)
    legacy = pd.read_csv(path)
    legacy = legacy[legacy["team"] != "__rho__"]
    legacy.to_csv(path, index=False)

    loaded = PoissonModel.load(path)
    assert loaded.rho == 0.0
    assert loaded.intercept == pytest.approx(model.intercept)


def test_load_rejects_csv_without_meta_rows(tmp_path):
    """A strengths CSV lacking the __intercept__/__home_advantage__ meta rows
    can't be turned into expected goals -- reject it with a clear error rather
    than a bare KeyError."""
    path = tmp_path / "no_meta.csv"
    pd.DataFrame(
        {"team": ["Strong"], "attack": [0.1], "defense": [0.1],
         "net_strength": [0.2]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required Poisson meta rows"):
        PoissonModel.load(path)


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
    # Distinct teams (unlike a self-match) so the winner assertion is a real
    # check that a knockout always resolves to one of the two sides.
    for _ in range(200):
        res = simulate_match(model, "Strong", "Weak", rng, knockout=True)
        assert not res.is_draw
        assert res.winner in ("Strong", "Weak")


class _AlwaysLevelModel:
    """Stub that forces a knockout tie all the way to penalties.

    Regulation always ends 1-1, and expected goals are 0 so extra time adds
    nothing -- so every knockout must be decided on penalties.
    """

    def sample_score(self, home, away, rng, neutral=True):
        return 1, 1

    def expected_goals(self, home, away, neutral=True):
        return 0.0, 0.0


def test_knockout_tie_resolves_via_extra_time_and_penalties():
    rng = np.random.default_rng(0)
    res = simulate_match(_AlwaysLevelModel(), "A", "B", rng, knockout=True)
    assert res.extra_time is True
    assert res.penalties is True
    assert not res.is_draw
    assert res.winner in ("A", "B")
    # Score stays level (1-1 after a scoreless extra time); pens pick a winner.
    assert res.home_score == res.away_score == 1


def test_unseen_team_warns_once(training_matches):
    model = PoissonModel().fit(training_matches)
    with pytest.warns(UserWarning, match="not seen in training data"):
        model.expected_goals("Atlantis", "Strong", neutral=True)
    # Repeat lookups for the same team must be silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.expected_goals("Atlantis", "Strong", neutral=True)


# --------------------------------------------------------- Dixon-Coles (3a)
def test_rho_within_bounds_on_real_data():
    """Fitting on the real 1930-2026 finals data finds a non-trivial rho."""
    matches = load_matches()
    training = build_training_matches(matches)
    model = PoissonModel().fit(training)
    lo, hi = model.params.rho_bounds
    assert lo <= model.rho <= hi
    assert model.rho != 0.0


def test_scoreline_matrix_dixon_coles_adjusts_only_low_score_cells(
    training_matches,
):
    model = PoissonModel().fit(training_matches)
    assert model.rho != 0.0  # sanity: the fit should find some correlation

    corrected = model.scoreline_matrix("Strong", "Weak", neutral=True)
    assert corrected.min() >= 0
    assert pytest.approx(corrected.sum(), abs=1e-9) == 1.0

    # Force rho to zero to get the plain independent-Poisson matrix, holding
    # every other fitted parameter fixed.
    model.rho = 0.0
    independent = model.scoreline_matrix("Strong", "Weak", neutral=True)

    max_g = model.params.max_goals
    low_score_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            if (i, j) in low_score_cells:
                continue
            assert corrected[i, j] == pytest.approx(independent[i, j], rel=1e-9)
    assert not np.allclose(
        [corrected[i, j] for i, j in low_score_cells],
        [independent[i, j] for i, j in low_score_cells],
    )


def test_sample_score_matches_scoreline_matrix(training_matches):
    """Empirical (0,0) frequency over many samples should track its cell in
    scoreline_matrix(), confirming sample_score draws from the corrected
    joint rather than independent margins."""
    model = PoissonModel().fit(training_matches)
    p00 = model.scoreline_matrix("Strong", "Weak", neutral=True)[0, 0]

    rng = np.random.default_rng(123)
    n = 20_000
    count00 = sum(
        model.sample_score("Strong", "Weak", rng, neutral=True) == (0, 0)
        for _ in range(n)
    )
    freq = count00 / n
    se = (p00 * (1.0 - p00) / n) ** 0.5
    assert abs(freq - p00) < 3 * se


def test_dixon_coles_disabled_keeps_rho_zero_and_independent_matrix(
    training_matches,
):
    config = Config(poisson=PoissonParams(dixon_coles=False))
    model = PoissonModel(config).fit(training_matches)
    assert model.rho == 0.0

    lam_h, lam_a = model.expected_goals("Strong", "Weak", neutral=True)
    goals = np.arange(model.params.max_goals + 1)
    expected = np.outer(
        poisson_dist.pmf(goals, lam_h), poisson_dist.pmf(goals, lam_a)
    )
    expected = expected / expected.sum()

    matrix = model.scoreline_matrix("Strong", "Weak", neutral=True)
    assert np.allclose(matrix, expected)
