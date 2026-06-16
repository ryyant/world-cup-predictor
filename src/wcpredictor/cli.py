"""Command-line interface for the World Cup 2026 predictor.

Commands
--------
    wcpredict train                 fit Elo + Poisson and save artifacts
    wcpredict ratings [--top N]     show the Elo ranking
    wcpredict match TEAM_A TEAM_B   predict a single fixture
    wcpredict simulate [--n N]      Monte Carlo the tournament
    wcpredict backtest              historical accuracy of the models
"""

from __future__ import annotations

import sys

import click

from wcpredictor.config import default_config
from wcpredictor.data.loader import load_groups, load_matches, load_teams
from wcpredictor.data.preprocess import build_training_matches
from wcpredictor.evaluation.metrics import backtest as run_backtest
from wcpredictor.models.elo import EloModel
from wcpredictor.models.poisson import PoissonModel
from wcpredictor.simulation.tournament import TournamentSimulator


def _load_training(config):
    matches = load_matches(config)
    return build_training_matches(matches, config)


def _load_or_train_elo(config) -> EloModel:
    if config.elo_ratings_path.exists():
        return EloModel.load(config.elo_ratings_path, config)
    click.echo("No saved Elo ratings found; training now...", err=True)
    return EloModel(config).fit(_load_training(config))


def _load_or_train_poisson(config) -> PoissonModel:
    if config.poisson_strengths_path.exists():
        return PoissonModel.load(config.poisson_strengths_path, config)
    click.echo("No saved Poisson model found; training now...", err=True)
    return PoissonModel(config).fit(_load_training(config))


@click.group()
@click.version_option(package_name="wcpredictor")
def cli():
    """Statistical World Cup 2026 predictor (Elo + Poisson)."""


@cli.command()
def train():
    """Fit Elo and Poisson models and save them to data/processed."""
    config = default_config()
    training = _load_training(config)
    teams = load_teams(config)

    click.echo(f"Training on {len(training)} matches...")
    elo = EloModel(config).fit(training)
    poisson = PoissonModel(config).fit(training)

    elo.save(config.elo_ratings_path, teams)
    poisson.save(config.poisson_strengths_path, teams)
    click.echo(f"Saved Elo ratings -> {config.elo_ratings_path}")
    click.echo(f"Saved Poisson strengths -> {config.poisson_strengths_path}")


@cli.command()
@click.option("--top", default=20, show_default=True, help="Number of teams.")
def ratings(top):
    """Show the current Elo ranking."""
    config = default_config()
    elo = _load_or_train_elo(config)
    try:
        teams = load_teams(config)
    except FileNotFoundError:
        teams = None
    table = elo.ratings_table(teams).head(top)
    click.echo(f"{'#':>3}  {'Team':<16}{'Rating':>8}")
    for i, row in enumerate(table.itertuples(index=False), start=1):
        click.echo(f"{i:>3}  {row.team:<16}{row.rating:>8.0f}")


@cli.command()
@click.argument("team_a")
@click.argument("team_b")
@click.option("--neutral/--home", default=True,
              help="Neutral venue (default) or TEAM_A at home.")
def match(team_a, team_b, neutral):
    """Predict a single fixture: TEAM_A vs TEAM_B."""
    config = default_config()
    elo = _load_or_train_elo(config)
    poisson = _load_or_train_poisson(config)

    elo_pred = elo.predict_match(team_a, team_b, neutral)
    poi_pred = poisson.predict_match(team_a, team_b, neutral)
    lam_a, lam_b = poisson.expected_goals(team_a, team_b, neutral)
    score_a, score_b = poisson.most_likely_score(team_a, team_b, neutral)

    venue = "neutral venue" if neutral else f"{team_a} at home"
    col_a = f"{team_a} win"
    col_b = f"{team_b} win"
    width = max(len(col_a), len(col_b)) + 3
    click.echo(f"\n{team_a} vs {team_b}  ({venue})\n" + "-" * (10 + 2 * width + 8))
    click.echo(f"{'Model':<10}{col_a:>{width}}{'Draw':>8}{col_b:>{width}}")
    click.echo(
        f"{'Elo':<10}{elo_pred.p_home_win:>{width}.1%}{elo_pred.p_draw:>8.1%}"
        f"{elo_pred.p_away_win:>{width}.1%}"
    )
    click.echo(
        f"{'Poisson':<10}{poi_pred.p_home_win:>{width}.1%}{poi_pred.p_draw:>8.1%}"
        f"{poi_pred.p_away_win:>{width}.1%}"
    )
    click.echo(
        f"\nExpected goals: {team_a} {lam_a:.2f} - {lam_b:.2f} {team_b}"
    )
    click.echo(f"Most likely score: {team_a} {score_a}-{score_b} {team_b}\n")


@cli.command()
@click.option("--n", "n_sim", default=None, type=int,
              help="Number of simulations (default from config).")
@click.option("--top", default=20, show_default=True,
              help="Number of teams to display.")
def simulate(n_sim, top):
    """Run the Monte Carlo tournament simulation."""
    config = default_config()
    poisson = _load_or_train_poisson(config)
    groups = load_groups(config)

    sim = TournamentSimulator(poisson, groups, config)
    n = n_sim or config.simulation.n_simulations
    click.echo(f"Simulating the tournament {n:,} times...")
    report = sim.run(n)

    table = report.table.head(top)
    click.echo(
        f"\n{'#':>3}  {'Team':<16}{'Grp':>4}{'Win':>8}{'Final':>8}"
        f"{'Semi':>8}{'Last8':>8}{'Adv':>8}"
    )
    for i, row in enumerate(table.itertuples(index=False), start=1):
        click.echo(
            f"{i:>3}  {row.team:<16}{row.group:>4}"
            f"{row.p_winner:>8.1%}{row.p_final:>8.1%}"
            f"{row.p_semifinal:>8.1%}{row.p_quarterfinal:>8.1%}"
            f"{row.p_advance:>8.1%}"
        )
    click.echo("")


@cli.command()
@click.option("--min-train", default=300, show_default=True,
              help="Matches used before scoring begins.")
def backtest(min_train):
    """Backtest Elo and Poisson on historical results."""
    config = default_config()
    training = _load_training(config)

    click.echo("Backtesting (walk-forward)...")
    elo_res = run_backtest(lambda: EloModel(config), training, config,
                           min_train=min_train)
    poi_res = run_backtest(lambda: PoissonModel(config), training, config,
                           min_train=min_train)
    click.echo(f"  Elo     {elo_res}")
    click.echo(f"  Poisson {poi_res}")
    click.echo(
        "\nLower log_loss / RPS is better; higher accuracy is better."
    )


def main():
    try:
        cli()
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
