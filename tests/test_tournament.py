"""Tests for the tournament simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredictor.models.poisson import PoissonModel
from wcpredictor.simulation.tournament import (
    TournamentSimulator,
    TeamStats,
    _bracket_seed_order,
)


def _make_groups() -> pd.DataFrame:
    """48 teams across 12 groups; team names encode a strength tier."""
    rows = []
    letters = [chr(ord("A") + i) for i in range(12)]
    n = 0
    for g in letters:
        for slot in range(1, 5):
            rows.append({"group": g, "slot": slot, "team": f"T{n:02d}"})
            n += 1
    return pd.DataFrame(rows)


def _make_model(groups: pd.DataFrame) -> PoissonModel:
    """A Poisson model whose strengths increase with team index."""
    model = PoissonModel()
    teams = list(groups["team"])
    model.attack = {t: 0.02 * i for i, t in enumerate(teams)}
    model.defense = {t: 0.02 * i for i, t in enumerate(teams)}
    model.intercept = np.log(1.3)
    model.home_advantage = np.log(1.1)
    model._teams = teams
    return model


def test_bracket_seed_order_keeps_top_seeds_apart():
    order = _bracket_seed_order(4)
    assert order == [1, 4, 2, 3]
    order8 = _bracket_seed_order(8)
    assert sorted(order8) == list(range(1, 9))
    # Seeds 1 and 2 are in opposite halves.
    assert order8.index(1) < 4 and order8.index(2) >= 4


def test_team_stats_points_and_gd():
    s = TeamStats(played=3, win=2, draw=1, loss=0, gf=5, ga=2)
    assert s.points == 7
    assert s.gd == 3


def test_group_ranking_orders_by_strength():
    groups = _make_groups()
    model = _make_model(groups)
    sim = TournamentSimulator(model, groups)
    rng = np.random.default_rng(0)
    # Strongest team in group A is T03 (highest index in slots 0-3).
    teams = sim.groups["A"]
    # Aggregate ranking over many sims: strongest should usually finish top.
    top_counts = {t: 0 for t in teams}
    for _ in range(200):
        ranking = sim.simulate_group(teams, rng)
        top_counts[ranking[0]] += 1
    assert max(top_counts, key=top_counts.get) == "T03"


def test_simulate_once_structure():
    groups = _make_groups()
    model = _make_model(groups)
    sim = TournamentSimulator(model, groups)
    rng = np.random.default_rng(1)
    result = sim.simulate_once(rng)
    assert len(result["qualifiers"]) == 32
    assert len(set(result["qualifiers"])) == 32
    assert len(result["reached"]["round_of_32"]) == 32
    assert len(result["reached"]["round_of_16"]) == 16
    assert len(result["reached"]["quarterfinal"]) == 8
    assert len(result["reached"]["semifinal"]) == 4
    assert len(result["reached"]["final"]) == 2
    assert len(result["reached"]["winner"]) == 1
    assert result["champion"] in result["reached"]["final"]


def test_run_probabilities_are_consistent():
    groups = _make_groups()
    model = _make_model(groups)
    sim = TournamentSimulator(model, groups)
    report = sim.run(n_simulations=200)
    table = report.table
    assert len(table) == 48
    # Winner probabilities sum to 1 (exactly one champion per sim).
    assert pytest.approx(table["p_winner"].sum(), abs=1e-9) == 1.0
    # Advance probabilities sum to 32 (32 qualifiers per sim).
    assert pytest.approx(table["p_advance"].sum(), abs=1e-6) == 32.0
    # Monotonic funnel: advancing >= reaching later rounds.
    for _, row in table.iterrows():
        assert row["p_advance"] >= row["p_round_of_16"] >= row["p_quarterfinal"]
        assert row["p_quarterfinal"] >= row["p_semifinal"] >= row["p_final"]
        assert row["p_final"] >= row["p_winner"]


def test_stronger_teams_win_more_often():
    groups = _make_groups()
    model = _make_model(groups)
    sim = TournamentSimulator(model, groups)
    report = sim.run(n_simulations=300)
    table = report.table.set_index("team")
    # T47 is the strongest team, T00 the weakest.
    assert table.loc["T47", "p_winner"] > table.loc["T00", "p_winner"]
