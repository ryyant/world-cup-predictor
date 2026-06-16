"""Tests for data loading, validation, and preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from wcpredictor.config import default_config
from wcpredictor.data.loader import (
    DataValidationError,
    load_groups,
    load_matches,
    load_teams,
    _require_columns,
)
from wcpredictor.data.preprocess import build_training_matches, team_match_counts


def test_bundled_data_loads():
    matches = load_matches()
    teams = load_teams()
    groups = load_groups()
    assert len(matches) > 0
    assert len(teams) == 48
    assert groups["group"].nunique() == 12
    assert (groups.groupby("group").size() == 4).all()


def test_all_group_teams_have_metadata():
    teams = set(load_teams()["team"])
    group_teams = set(load_groups()["team"])
    assert group_teams <= teams


def test_require_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError):
        _require_columns(df, {"a", "b"}, default_config().matches_path)


def test_build_training_matches_adds_fields():
    matches = load_matches()
    tr = build_training_matches(matches)
    assert {"result", "goal_diff", "total_goals", "recency_weight"} <= set(
        tr.columns
    )
    assert tr["result"].isin({"H", "D", "A"}).all()
    assert (tr["recency_weight"] > 0).all()
    assert tr["recency_weight"].max() <= 1.0 + 1e-9


def test_team_match_counts_positive():
    counts = team_match_counts(load_matches())
    assert (counts > 0).all()
