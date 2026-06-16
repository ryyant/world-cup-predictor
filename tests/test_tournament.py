"""Tests for the tournament simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredictor.data.tournament_state import KnockoutMatch, TournamentState
from wcpredictor.simulation.tournament import (
    SimulationReport,
    TournamentSimulator,
    TeamStats,
    _bracket_seed_order,
)


def _qf_state(groups48) -> TournamentState:
    """A synthetic 'we're at the quarterfinals' state over the 48 test teams.

    The eight strongest teams (T40-T47) are alive; T38/T39 model teams knocked
    out in the R16 and R32; everyone else went out in the group stage. Group
    finishes mirror each team's slot so every group is a clean 1-4.
    """
    teams = list(groups48["team"])
    alive = teams[-8:]  # T40..T47
    frontier = tuple(
        KnockoutMatch(alive[i], alive[i + 1]) for i in range(0, 8, 2)
    )
    reached = {t: "quarterfinal" for t in alive}
    reached["T39"] = "round_of_16"
    reached["T38"] = "round_of_32"
    group_position = {
        row.team: int(row.slot) for row in groups48.itertuples(index=False)
    }
    return TournamentState(
        reached=reached,
        group_position=group_position,
        frontier_stage="quarterfinal",
        frontier=frontier,
    )


def test_bracket_seed_order_keeps_top_seeds_apart():
    order = _bracket_seed_order(4)
    assert order == [1, 4, 2, 3]
    order8 = _bracket_seed_order(8)
    assert sorted(order8) == list(range(1, 9))
    # Seeds 1 and 2 are in opposite halves.
    assert order8.index(1) < 4 and order8.index(2) >= 4


def test_team_stats_points_and_gd():
    s = TeamStats(win=2, draw=1, loss=0, gf=5, ga=2)
    assert s.points == 7
    assert s.gd == 3
    assert s.played == 3  # derived from win + draw + loss


def test_group_ranking_orders_by_strength(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    rng = np.random.default_rng(0)
    # Strongest team in group A is T03 (highest index in slots 0-3).
    teams = sim.groups["A"]
    # Aggregate ranking over many sims: strongest should usually finish top.
    top_counts = {t: 0 for t in teams}
    for _ in range(200):
        ranking = sim.simulate_group(teams, rng)
        top_counts[ranking[0]] += 1
    assert max(top_counts, key=top_counts.get) == "T03"


def test_simulate_once_structure(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
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


def test_run_probabilities_are_consistent(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
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


def test_run_rejects_non_positive_n(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    for bad in (0, -5):
        with pytest.raises(ValueError, match="must be positive"):
            sim.run(n_simulations=bad)


def test_simulator_requires_twelve_groups(tiered_model):
    # Eight groups of 4 -> not the 12/12/8 structure the bracket needs; must
    # fail fast, not IndexError deep in the knockout.
    rows = []
    for i, g in enumerate("ABCDEFGH"):
        for slot in range(1, 5):
            rows.append({"group": g, "slot": slot, "team": f"X{i}{slot}"})
    with pytest.raises(ValueError, match="exactly 12 groups"):
        TournamentSimulator(tiered_model, pd.DataFrame(rows))


def test_stronger_teams_win_more_often(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    report = sim.run(n_simulations=300)
    table = report.table.set_index("team")
    # T47 is the strongest team, T00 the weakest.
    assert table.loc["T47", "p_winner"] > table.loc["T00", "p_winner"]


# ----------------------------------------------------- standard_errors (3c)
def test_standard_errors_match_binomial_formula(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    report = sim.run(n_simulations=200)
    se_table = report.standard_errors()

    # Identifying columns carried over unchanged, table itself untouched.
    assert list(se_table["team"]) == list(report.table["team"])
    assert list(se_table["group"]) == list(report.table["group"])
    assert "se_winner" not in report.table.columns

    # Spot-check the formula directly on a couple of columns.
    for p_col, se_col in [("p_winner", "se_winner"), ("p_advance", "se_advance")]:
        p = report.table[p_col]
        expected = np.sqrt(p * (1.0 - p) / report.n_simulations)
        assert np.allclose(se_table[se_col], expected)


def test_standard_errors_zero_at_extreme_probabilities():
    table = pd.DataFrame(
        {"team": ["A", "B"], "group": ["X", "X"], "p_winner": [0.0, 1.0]}
    )
    report = SimulationReport(table=table, n_simulations=100)
    se = report.standard_errors()
    assert se["se_winner"].tolist() == [0.0, 0.0]


# ---------------------------------------------------- conditioned simulation
def test_run_conditioned_only_alive_teams_can_win(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    state = _qf_state(groups48)
    report = sim.run_conditioned(state, n_simulations=500)
    table = report.table.set_index("team")

    assert len(table) == 48
    # Exactly one champion per sim, and only the eight alive teams can be it.
    assert pytest.approx(table["p_winner"].sum(), abs=1e-9) == 1.0
    winners = set(table.index[table["p_winner"] > 0])
    assert winners == set(state.alive)


def test_run_conditioned_locks_in_settled_rounds(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    state = _qf_state(groups48)
    table = sim.run_conditioned(state, n_simulations=300).table.set_index("team")

    # Alive team: reached the QF for certain, still uncertain beyond it.
    assert table.loc["T47", "p_advance"] == 1.0
    assert table.loc["T47", "p_quarterfinal"] == 1.0
    assert 0.0 < table.loc["T47", "p_winner"] < 1.0

    # Eliminated in the R16: reached R16 for certain, no chance beyond it.
    assert table.loc["T39", "p_advance"] == 1.0
    assert table.loc["T39", "p_round_of_16"] == 1.0
    assert table.loc["T39", "p_quarterfinal"] == 0.0
    assert table.loc["T39", "p_winner"] == 0.0

    # Out in the group stage: never even advanced.
    assert table.loc["T00", "p_advance"] == 0.0

    # Group finishes are settled facts (0/1), matching each team's slot.
    assert table.loc["T00", "p_group_1st"] == 1.0  # slot 1 in group A
    assert table.loc["T01", "p_group_2nd"] == 1.0  # slot 2 in group A


def test_run_conditioned_funnel_is_monotonic(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    report = sim.run_conditioned(_qf_state(groups48), n_simulations=400)
    for _, row in report.table.iterrows():
        assert row["p_advance"] >= row["p_round_of_16"] >= row["p_quarterfinal"]
        assert row["p_quarterfinal"] >= row["p_semifinal"] >= row["p_final"]
        assert row["p_final"] >= row["p_winner"]


def test_run_conditioned_stronger_alive_team_wins_more(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    table = sim.run_conditioned(
        _qf_state(groups48), n_simulations=1500
    ).table.set_index("team")
    # T47 is the strongest team left; T40 the weakest of the eight alive.
    assert table.loc["T47", "p_winner"] > table.loc["T40", "p_winner"]


def test_run_conditioned_rejects_non_positive_n(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    state = _qf_state(groups48)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="must be positive"):
            sim.run_conditioned(state, n_simulations=bad)


def test_simulate_once_conditioned_structure(groups48, tiered_model):
    sim = TournamentSimulator(tiered_model, groups48)
    state = _qf_state(groups48)
    rng = np.random.default_rng(3)
    result = sim.simulate_once_conditioned(state, rng)

    assert len(result["reached"]["quarterfinal"]) == 8
    assert set(state.alive) <= result["reached"]["quarterfinal"]
    assert len(result["reached"]["semifinal"]) == 4
    assert len(result["reached"]["final"]) == 2
    assert len(result["reached"]["winner"]) == 1
    assert result["champion"] in result["reached"]["final"]
    assert result["champion"] in state.alive
