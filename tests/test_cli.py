"""Tests for the ``wcpredict`` command-line interface.

These exercise the CLI end-to-end against the bundled data via click's
``CliRunner``. Commands that need a model load a saved artifact if one exists
under ``data/processed`` or otherwise train in memory, so the tests pass in a
clean checkout as well as after ``wcpredict train``. None of the commands
tested here write files.
"""

from __future__ import annotations

from click.testing import CliRunner

from wcpredictor.cli import cli


def _run(*args):
    return CliRunner().invoke(cli, list(args))


def test_version_option():
    result = _run("--version")
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_ratings_lists_teams():
    result = _run("ratings", "--top", "5")
    assert result.exit_code == 0, result.output
    assert "Team" in result.output
    assert "Rating" in result.output


def test_match_reports_both_models():
    result = _run("match", "Brazil", "Argentina")
    assert result.exit_code == 0, result.output
    assert "Elo" in result.output
    assert "Poisson" in result.output
    assert "Expected goals" in result.output


def test_simulate_conditions_on_played_results_by_default():
    result = _run("simulate", "--n", "12", "--top", "5")
    assert result.exit_code == 0, result.output
    # The vendored 2026 tournament is complete, so the default view has no
    # frontier left; simulate rewinds to the final and still conditions on the
    # settled results rather than replaying the whole event from scratch.
    assert "Conditioning on results" in result.output
    # The Monte Carlo standard-error footer must still render.
    assert "Monte Carlo" in result.output
    assert "s.e." in result.output


def test_simulate_from_scratch_projects_whole_tournament():
    result = _run("simulate", "--n", "12", "--top", "5", "--from-scratch")
    assert result.exit_code == 0, result.output
    assert "Simulating the tournament 12 times" in result.output
    assert "Monte Carlo" in result.output


def test_simulate_rejects_non_positive_n():
    result = _run("simulate", "--n", "0")
    assert result.exit_code != 0
    # Rejected up front, before any simulation runs.
    assert "Simulating the tournament" not in result.output


def test_backtest_reports_both_models():
    # A high min_train keeps the walk-forward short (few refit blocks) so the
    # test stays fast while still touching the real backtest path.
    result = _run("backtest", "--min-train", "900")
    assert result.exit_code == 0, result.output
    assert "Elo" in result.output
    assert "Poisson" in result.output
