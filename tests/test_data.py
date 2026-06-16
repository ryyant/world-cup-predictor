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
    _parse_neutral,
    _require_columns,
)
from wcpredictor.data.preprocess import (
    build_training_matches,
    recency_weights,
    team_match_counts,
)

_MATCH_HEADER = "date,home_team,away_team,home_score,away_score,neutral,tournament"
_GROUP_HEADER = "group,slot,team"


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


# --------------------------------------------------------- neutral parsing
def test_parse_neutral_recognizes_tokens():
    parsed = _parse_neutral(
        pd.Series(["1", "0", "true", "False", "yes", "n"]),
        default_config().matches_path,
    )
    assert parsed.tolist() == [True, False, True, False, True, False]


def test_parse_neutral_rejects_missing_or_unknown():
    # A NaN cell used to coerce to True via astype(bool), silently marking a
    # home game neutral; now it must raise instead of guessing.
    for bad in (pd.Series([1, None]), pd.Series(["1", ""]), pd.Series(["maybe"])):
        with pytest.raises(DataValidationError, match="neutral"):
            _parse_neutral(bad, default_config().matches_path)


def test_load_matches_rejects_blank_neutral(tmp_path):
    path = tmp_path / "matches.csv"
    path.write_text(
        f"{_MATCH_HEADER}\n"
        "1930-07-13,France,Mexico,4,1,1,World Cup 1930\n"
        "1930-07-14,Brazil,Peru,2,1,,World Cup 1930\n"  # blank neutral
    )
    config = default_config().with_overrides(raw_data_dir=tmp_path)
    with pytest.raises(DataValidationError, match="neutral"):
        load_matches(config)


# ------------------------------------------------------ group-count validation
def test_load_groups_rejects_wrong_group_count(tmp_path):
    lines = [_GROUP_HEADER]
    for g in "ABCDEFGH":  # only 8 groups
        for slot in range(1, 5):
            lines.append(f"{g},{slot},{g}{slot}")
    (tmp_path / default_config().groups_csv).write_text("\n".join(lines) + "\n")
    config = default_config().with_overrides(raw_data_dir=tmp_path)
    with pytest.raises(DataValidationError, match="12 groups"):
        load_groups(config)


# ----------------------------------------------------------- recency weights
def test_recency_weights_reference_date_anchors_scale():
    dates = pd.to_datetime(pd.Series(["2000-01-01", "2010-01-01", "2020-01-01"]))
    config = default_config()

    # Anchored to the latest date: most recent match weighted 1.0.
    w_latest = recency_weights(dates, config)
    assert w_latest[-1] == pytest.approx(1.0)
    assert (w_latest[:-1] < 1.0).all()

    # Anchored to an earlier cutoff: the match on that cutoff gets weight 1.0,
    # and the whole vector is NOT just a uniform rescale of the other one
    # (this is what the backtest relies on to avoid over-shrinking).
    w_cutoff = recency_weights(
        dates, config, reference_date=pd.Timestamp("2010-01-01")
    )
    assert w_cutoff[1] == pytest.approx(1.0)
