"""Tests for outcome-distribution derivation, group positions, and viz."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from wcpredictor.evaluation import BacktestResult
from wcpredictor.models.elo import EloModel
from wcpredictor.simulation.tournament import EXACT_OUTCOMES, TournamentSimulator
from wcpredictor import visualization as viz


@pytest.fixture(scope="module")
def report(groups48, tiered_model):
    return TournamentSimulator(tiered_model, groups48).run(n_simulations=300)


def test_outcome_distribution_sums_to_one(report):
    dist = report.outcome_distribution()
    keys = [k for k, _ in EXACT_OUTCOMES]
    assert set(keys).issubset(dist.columns)
    row_sums = dist[keys].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)
    assert (dist[keys] >= 0).all().all()


def test_champion_equals_p_winner(report):
    dist = report.outcome_distribution().set_index("team")
    table = report.table.set_index("team")
    assert np.allclose(dist["champion"], table["p_winner"])


def test_group_positions_sum_to_one(report):
    gp = report.group_position_distribution()
    cols = ["p_group_1st", "p_group_2nd", "p_group_3rd", "p_group_4th"]
    assert np.allclose(gp[cols].sum(axis=1), 1.0, atol=1e-9)


def test_stronger_team_more_likely_to_top_group(report):
    gp = report.group_position_distribution().set_index("team")
    # In group A (T00..T03), T03 is strongest and should top it most often.
    group_a = [t for t in gp.index if t in {"T00", "T01", "T02", "T03"}]
    top = gp.loc[group_a, "p_group_1st"].idxmax()
    assert top == "T03"


def test_outcome_plots_return_axes(report):
    assert viz.plot_outcome_distribution(report, top_n=8) is not None
    assert viz.plot_stage_heatmap(report, top_n=8) is not None
    assert viz.plot_title_race(report, top_n=8) is not None
    group = sorted(report.table["group"].unique())[0]
    assert viz.plot_group_outcomes(report, group) is not None
    fig = viz.plot_group_grid(report)
    assert fig is not None
    plt.close("all")


def test_match_and_scoreline_plots(tiered_model):
    pred = tiered_model.predict_match("T47", "T00", neutral=True)
    ax = viz.plot_match_comparison({"Poisson": pred}, "T47", "T00")
    assert ax is not None
    ax2 = viz.plot_scoreline_heatmap(tiered_model, "T47", "T00", max_goals=5)
    assert ax2 is not None
    plt.close("all")


# --------------------------------------------------------- calibration (4c)
def test_plot_calibration_returns_axes():
    calib = pd.DataFrame(
        {
            "outcome": ["H", "H", "D", "A"],
            "bin_mid": [0.25, 0.75, 0.5, 0.5],
            "mean_predicted": [0.24, 0.74, 0.51, 0.49],
            "empirical_freq": [0.20, 0.70, 0.55, 0.45],
            "count": [40, 60, 30, 30],
        }
    )
    ax = viz.plot_calibration(calib)
    assert ax is not None
    plt.close("all")


def test_plot_rating_history_returns_axes():
    history = pd.DataFrame(
        {
            "date": [
                "2020-01-01", "2020-02-01", "2020-03-01", "2020-01-15",
            ],
            "team": ["Alpha", "Alpha", "Alpha", "Beta"],
            "rating": [1500.0, 1520.0, 1535.0, 1480.0],
        }
    )
    ax = viz.plot_rating_history(history, ["Alpha", "Beta"])
    assert ax is not None
    plt.close("all")


def test_plot_model_comparison_returns_axes():
    results = {
        "Elo": BacktestResult(accuracy=0.50, log_loss=1.02, rps=0.21, n_matches=100),
        "Poisson": BacktestResult(accuracy=0.55, log_loss=0.95, rps=0.18, n_matches=100),
    }
    ax = viz.plot_model_comparison(results)
    assert ax is not None
    plt.close("all")


def test_plot_group_difficulty_returns_axes(groups48):
    elo = EloModel()
    elo.ratings = {team: 1500.0 + i for i, team in enumerate(groups48["team"])}
    ax = viz.plot_group_difficulty(groups48, elo)
    assert ax is not None
    plt.close("all")


def test_title_race_show_se_variants(report):
    ax_with_se = viz.plot_title_race(report, top_n=8, show_se=True)
    assert ax_with_se is not None
    ax_without_se = viz.plot_title_race(report, top_n=8, show_se=False)
    assert ax_without_se is not None
    plt.close("all")
