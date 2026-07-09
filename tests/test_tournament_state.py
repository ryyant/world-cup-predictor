"""Tests for parsing the played 2026 results into a conditionable state."""

from __future__ import annotations

import json
import math

import pytest

from wcpredictor.config import default_config
from wcpredictor.data import (
    actual_knockout_ties,
    load_groups,
    load_tournament_state,
    phase_start_dates,
)
from wcpredictor.data.tournament_state import (
    GROUP_STAGE,
    TournamentStateError,
    load_tournament_state as load_state,
)
from wcpredictor.simulation.tournament import STAGES, _STAGE_INDEX


def _write(tmp_path, matches):
    """Write a minimal 2026 fixtures doc and return a config pointing at it."""
    path = tmp_path / "2026.worldcup.json"
    path.write_text(json.dumps({"name": "test", "matches": matches}),
                    encoding="utf-8")
    return default_config().with_overrides(source_data_dir=tmp_path)


def _group_round_robin(letter, teams, scores):
    """Six group matches for a 4-team group with the given (h, a) scorelines."""
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
    out = []
    for (i, j), (hg, ag) in zip(pairs, scores):
        out.append({
            "round": "Matchday 1", "group": f"Group {letter}",
            "team1": teams[i], "team2": teams[j],
            "score": {"ft": [hg, ag]},
        })
    return out


# --------------------------------------------------------- the real fixtures
def test_real_state_is_internally_consistent():
    """The bundled 2026 file must parse into a coherent, playable frontier.

    Written to survive the tournament progressing (frontier moving from the
    QF to the SF, etc.) by asserting structural invariants, not a fixed round.
    """
    config = default_config()
    state = load_tournament_state(config)

    # The frontier is a real, not-yet-won round.
    assert state.frontier_stage in STAGES
    assert state.frontier_stage != "winner"

    alive = state.alive
    assert len(alive) == 2 * len(state.frontier)  # two teams per tie
    assert len(set(alive)) == len(alive)          # no team twice
    # A single-elimination bracket has a power-of-two field.
    assert len(alive) & (len(alive) - 1) == 0
    # Winners march through exactly log2(field) more stages, ending in "winner".
    assert len(state.remaining_stages) == int(math.log2(len(alive)))
    assert state.remaining_stages[-1] == "winner"
    # remaining stages are the tail of STAGES immediately after the frontier.
    assert state.remaining_stages == STAGES[_STAGE_INDEX[state.frontier_stage] + 1:]

    # Every alive team has reached (only) the frontier stage.
    for team in alive:
        assert state.reached[team] == state.frontier_stage

    # reached values are valid stages; the alive set is drawn from the 48.
    draw = set(load_groups(config)["team"])
    assert set(alive) <= draw
    for team, stage in state.reached.items():
        assert stage in STAGES
        assert team in draw

    # Group finishes are a clean 1-4 permutation within each of the 12 groups.
    groups = load_groups(config)
    for _letter, sub in groups.groupby("group"):
        finishes = sorted(state.group_position[t] for t in sub["team"])
        assert finishes == [1, 2, 3, 4]


def test_real_state_eliminated_teams_are_capped():
    """Teams that already lost keep their reached stage but drop off later ones."""
    state = load_tournament_state(default_config())
    frontier_idx = _STAGE_INDEX[state.frontier_stage]
    # Anyone not alive must have reached a stage strictly before the frontier
    # (or never made the knockout at all).
    for team, stage in state.reached.items():
        if team not in state.alive:
            assert _STAGE_INDEX[stage] < frontier_idx


# ------------------------------------------------------------ synthetic docs
def test_parses_minimal_semifinal_bracket(tmp_path):
    matches = [
        {"num": 1, "round": "Semi-final", "team1": "Alpha", "team2": "Beta"},
        {"num": 2, "round": "Semi-final", "team1": "Gamma", "team2": "Delta"},
        {"num": 3, "round": "Final", "team1": "W1", "team2": "W2"},
    ]
    state = load_state(_write(tmp_path, matches))
    assert state.frontier_stage == "semifinal"
    assert state.alive == ("Alpha", "Beta", "Gamma", "Delta")
    assert state.remaining_stages == ("final", "winner")
    for t in state.alive:
        assert state.reached[t] == "semifinal"


def test_group_positions_ranked_by_points(tmp_path):
    # A beats everyone, B beats C and D, C beats D -> A,B,C,D order.
    scores = [(3, 0),   # A v B
              (1, 0),   # C v D
              (2, 0),   # A v C
              (2, 0),   # B v D
              (4, 0),   # A v D
              (1, 0)]   # B v C
    matches = _group_round_robin("A", ["A", "B", "C", "D"], scores)
    # A knockout match is required, else the parser reports "not started".
    matches.append({"num": 49, "round": "Round of 32",
                    "team1": "A", "team2": "Z", "score": {"ft": [1, 0]}})
    matches.append({"num": 81, "round": "Round of 16",
                    "team1": "A", "team2": "Y"})  # frontier
    state = load_state(_write(tmp_path, matches))
    assert state.group_position["A"] == 1
    assert state.group_position["B"] == 2
    assert state.group_position["C"] == 3
    assert state.group_position["D"] == 4


def test_penalty_tie_resolved_by_who_advanced(tmp_path):
    # Alpha 1-1 Beta in the R32 (a shootout); Alpha turns up in the R16, so
    # Alpha is recorded as having advanced and Beta as eliminated there.
    matches = [
        {"num": 1, "round": "Round of 32", "team1": "Alpha", "team2": "Beta",
         "score": {"ft": [1, 1]}},
        {"num": 33, "round": "Round of 16", "team1": "Alpha", "team2": "Gamma"},
    ]
    state = load_state(_write(tmp_path, matches))
    assert state.reached["Alpha"] == "round_of_16"
    assert state.reached["Beta"] == "round_of_32"
    assert "Beta" not in state.alive
    assert state.reached.get("Beta") != GROUP_STAGE  # it did reach the R32


def test_raises_on_partially_played_group_stage(tmp_path):
    scores = [(3, 0), (1, 0), (2, 0), (2, 0), (4, 0), (1, 0)]
    matches = _group_round_robin("A", ["A", "B", "C", "D"], scores)
    matches.append({"round": "Matchday 3", "group": "Group B",
                    "team1": "E", "team2": "F"})  # no score -> unplayed
    with pytest.raises(TournamentStateError, match="unplayed"):
        load_state(_write(tmp_path, matches))


def test_raises_when_knockout_not_started(tmp_path):
    scores = [(3, 0), (1, 0), (2, 0), (2, 0), (4, 0), (1, 0)]
    matches = _group_round_robin("A", ["A", "B", "C", "D"], scores)
    with pytest.raises(TournamentStateError, match="not reached the knockout"):
        load_state(_write(tmp_path, matches))


def test_raises_when_already_decided(tmp_path):
    matches = [
        {"num": 1, "round": "Final", "team1": "Alpha", "team2": "Beta",
         "score": {"ft": [2, 1]}},
    ]
    with pytest.raises(TournamentStateError, match="already decided"):
        load_state(_write(tmp_path, matches))


# --------------------------------------------------------- as_of_stage rewind
def test_as_of_stage_rewinds_frontier(tmp_path):
    """`as_of_stage` unplays a round (and everything after), moving the frontier.

    R32 and R16 are both played and the QF has real teams; rewinding to the R16
    should reopen it as the frontier and forget the QF matchup entirely.
    """
    matches = [
        {"num": 1, "round": "Round of 32", "team1": "Alpha", "team2": "Beta",
         "score": {"ft": [2, 0]}},
        {"num": 2, "round": "Round of 32", "team1": "Gamma", "team2": "Delta",
         "score": {"ft": [1, 0]}},
        {"num": 33, "round": "Round of 16", "team1": "Alpha", "team2": "Gamma",
         "score": {"ft": [1, 0]}},
        {"num": 49, "round": "Quarter-final", "team1": "Alpha", "team2": "Zeta"},
    ]
    config = _write(tmp_path, matches)

    # Default view: the QF is the frontier (R16 is already settled).
    assert load_state(config).frontier_stage == "quarterfinal"

    state = load_state(config, as_of_stage="round_of_16")
    assert state.frontier_stage == "round_of_16"
    assert state.alive == ("Alpha", "Gamma")
    assert state.remaining_stages == STAGES[_STAGE_INDEX["round_of_16"] + 1:]
    # Both survivors are back to only having reached the R16 (not the QF).
    assert state.reached["Alpha"] == "round_of_16"
    assert state.reached["Gamma"] == "round_of_16"
    # Losers keep the round they actually reached.
    assert state.reached["Beta"] == "round_of_32"
    # The QF opponent is forgotten -- that matchup isn't known yet at this point.
    assert "Zeta" not in state.reached


@pytest.mark.parametrize("bad", ["group_stage", "winner", "nonsense"])
def test_as_of_stage_rejects_non_knockout(tmp_path, bad):
    matches = [
        {"num": 1, "round": "Round of 32", "team1": "Alpha", "team2": "Beta",
         "score": {"ft": [2, 0]}},
        {"num": 33, "round": "Round of 16", "team1": "Alpha", "team2": "Gamma"},
    ]
    with pytest.raises(ValueError, match="as_of_stage"):
        load_state(_write(tmp_path, matches), as_of_stage=bad)


# ---------------------------------------------------- phase dates & actual ties
def test_phase_start_dates_are_chronological():
    """Each phase of the real 2026 fixtures starts no earlier than the last."""
    dates = phase_start_dates(default_config())
    assert dates[GROUP_STAGE]  # group stage is present and dated
    ordered = [GROUP_STAGE] + [s for s in STAGES if s != "winner"]
    present = [dates[s] for s in ordered if s in dates]
    assert present == sorted(present)


def test_actual_ties_align_with_rewound_frontier():
    """The real ties for a round equal the frontier we get by rewinding to it."""
    config = default_config()
    ties = actual_knockout_ties(config)
    assert "round_of_16" in ties
    # Played rounds have a recorded winner for every tie.
    assert all(t.winner for t in ties["round_of_16"])

    state = load_state(config, as_of_stage="round_of_16")
    actual_pairs = {frozenset((t.team1, t.team2)) for t in ties["round_of_16"]}
    frontier_pairs = {frozenset((m.team1, m.team2)) for m in state.frontier}
    assert frontier_pairs == actual_pairs
    # Each recorded winner is one of the two teams in its tie.
    for t in ties["round_of_16"]:
        assert t.winner in (t.team1, t.team2)
