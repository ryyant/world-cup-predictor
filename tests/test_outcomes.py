"""Tests for outcome-distribution derivation, group positions, and viz."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from wcpredictor.models.poisson import PoissonModel
from wcpredictor.simulation.tournament import EXACT_OUTCOMES, TournamentSimulator
from wcpredictor import visualization as viz


def _make_groups() -> pd.DataFrame:
    rows = []
    letters = [chr(ord("A") + i) for i in range(12)]
    n = 0
    for g in letters:
        for slot in range(1, 5):
            rows.append({"group": g, "slot": slot, "team": f"T{n:02d}"})
            n += 1
    return pd.DataFrame(rows)


def _make_model(groups: pd.DataFrame) -> PoissonModel:
    model = PoissonModel()
    teams = list(groups["team"])
    model.attack = {t: 0.02 * i for i, t in enumerate(teams)}
    model.defense = {t: 0.02 * i for i, t in enumerate(teams)}
    model.intercept = np.log(1.3)
    model.home_advantage = np.log(1.1)
    model._teams = teams
    return model


@pytest.fixture(scope="module")
def report():
    groups = _make_groups()
    model = _make_model(groups)
    return TournamentSimulator(model, groups).run(n_simulations=300)


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


def test_match_and_scoreline_plots():
    groups = _make_groups()
    model = _make_model(groups)
    pred = model.predict_match("T47", "T00", neutral=True)
    ax = viz.plot_match_comparison({"Poisson": pred}, "T47", "T00")
    assert ax is not None
    ax2 = viz.plot_scoreline_heatmap(model, "T47", "T00", max_goals=5)
    assert ax2 is not None
    plt.close("all")
