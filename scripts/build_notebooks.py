"""Generate the analysis notebooks under ``notebooks/``.

Building them programmatically with ``nbformat`` keeps the JSON valid and makes
the notebook contents easy to review and regenerate. Run:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def build(name: str, cells) -> None:
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    path = NOTEBOOK_DIR / name
    nbf.write(nb, path)
    print(f"wrote {path}")


MINIMAL_SETUP = """\
import matplotlib.pyplot as plt
import pandas as pd

from wcpredictor.config import default_config

config = default_config()
plt.rcParams["figure.figsize"] = (9, 4.5)
"""

# Shared notebook-only hex colors, kept in one place so they stay in sync with
# the win/draw/loss and accent language used throughout wcpredictor.visualization.
NB_PALETTE = {
    "home": "#2980b9",   # same blue as visualization.plot_match_comparison's home-win bar
    "draw": "#95a5a6",   # same grey as plot_match_comparison's draw bar
    "away": "#c0392b",   # same red as plot_match_comparison's away-win bar
    "accent": "#27ae60",
}


def notebook_01():
    return [
        new_markdown_cell(
            "# 01 - Data Exploration\n\n"
            "Inspect the bundled seed data (historical international results, "
            "team metadata, and the 2026 group draw). The match results are "
            "real World Cup finals results from openfootball, spanning 1930-2026. "
            "A handful of 2026 debutant nations have no historical World Cup "
            "finals matches; the models fall back to league-average strengths "
            "for them (with a runtime warning). The printed list below shows "
            "exactly which teams that applies to for the currently vendored data."
        ),
        new_code_cell(
            MINIMAL_SETUP
            + "\nfrom wcpredictor.data import load_matches, load_teams, load_groups\n"
            "from wcpredictor.data.preprocess import build_training_matches, team_match_counts"
        ),
        new_code_cell(
            "matches = load_matches(config)\n"
            "teams = load_teams(config)\n"
            "groups = load_groups(config)\n"
            "print('matches:', matches.shape)\n"
            "print('teams:', teams.shape)\n"
            "print('date range:', matches['date'].min().date(), '->', "
            "matches['date'].max().date())\n"
            "matches.head()"
        ),
        new_code_cell(
            "historical_teams = set(matches['home_team']) | set(matches['away_team'])\n"
            "debutants = sorted(set(groups['team']) - historical_teams)\n"
            "print('2026 teams with no historical World Cup finals data:', debutants)"
        ),
        new_markdown_cell("## Match outcomes (home / draw / away)"),
        new_code_cell(
            "tr = build_training_matches(matches, config)\n"
            "counts = tr['result'].value_counts().reindex(['H', 'D', 'A'])\n"
            f"ax = counts.plot.bar(color=[{NB_PALETTE['home']!r}, "
            f"{NB_PALETTE['draw']!r}, {NB_PALETTE['away']!r}])\n"
            "ax.set_title('Outcome distribution (home perspective)')\n"
            "ax.set_xlabel('result'); ax.set_ylabel('matches')\n"
            "plt.tight_layout(); plt.show()\n"
            "counts"
        ),
        new_markdown_cell("## Goals per match"),
        new_code_cell(
            "ax = tr['total_goals'].plot.hist(bins=range(0, 12), rwidth=0.9)\n"
            "ax.set_title('Total goals per match')\n"
            "ax.set_xlabel('goals'); plt.tight_layout(); plt.show()\n"
            "print('mean goals/match:', round(tr['total_goals'].mean(), 2))"
        ),
        new_markdown_cell("## Matches played per team and the group draw"),
        new_code_cell(
            "display(team_match_counts(matches).head(10).to_frame('matches'))\n"
            "groups.groupby('group')['team'].apply(list)"
        ),
    ]


def notebook_02():
    return [
        new_markdown_cell(
            "# 02 - Elo Ratings\n\n"
            "Fit the Elo model by replaying matches chronologically, then "
            "inspect the resulting ratings and a sample prediction."
        ),
        new_code_cell(
            MINIMAL_SETUP
            + "\nfrom wcpredictor.data import load_matches, load_teams\n"
            "from wcpredictor.data.preprocess import build_training_matches\n"
            "from wcpredictor.models import EloModel"
        ),
        new_code_cell(
            "tr = build_training_matches(load_matches(config), config)\n"
            "teams = load_teams(config)\n"
            "elo = EloModel(config).fit(tr, track_history=True)\n"
            "table = elo.ratings_table(teams)\n"
            "table.head(20)"
        ),
        new_markdown_cell("## Top 15 teams by Elo"),
        new_code_cell(
            "top = table.head(15).iloc[::-1]\n"
            "ax = top.plot.barh(x='team', y='rating', legend=False, "
            f"color={NB_PALETTE['accent']!r})\n"
            "ax.set_title('Top 15 Elo ratings'); ax.set_xlabel('rating')\n"
            "ax.set_xlim(table['rating'].min() - 20, table['rating'].max() + 20)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell("## Average rating by confederation"),
        new_code_cell(
            "by_conf = table.groupby('confederation')['rating']"
            ".mean().sort_values(ascending=False)\n"
            f"ax = by_conf.plot.bar(color={NB_PALETTE['accent']!r})\n"
            "ax.set_title('Mean Elo rating by confederation')\n"
            "ax.set_ylabel('rating'); plt.tight_layout(); plt.show()\n"
            "by_conf"
        ),
        new_markdown_cell("## Sample prediction"),
        new_code_cell(
            "pred = elo.predict_match('Brazil', 'Germany', neutral=True)\n"
            "print(f'Brazil win: {pred.p_home_win:.1%}')\n"
            "print(f'Draw:       {pred.p_draw:.1%}')\n"
            "print(f'Germany win:{pred.p_away_win:.1%}')"
        ),
        new_markdown_cell(
            "## Rating evolution\n\n"
            "Because the training data now spans real World Cup history "
            "(1930-2026) rather than a handful of synthetic years, we can "
            "watch these traditional powers' Elo ratings rise and fall "
            "across the decades."
        ),
        new_code_cell(
            "from wcpredictor.visualization import plot_rating_history\n"
            "plot_rating_history(elo.history, ['Brazil', 'Germany', 'Argentina', "
            "'France', 'Italy', 'England'])\n"
            "plt.tight_layout(); plt.show()"
        ),
    ]


def notebook_03():
    return [
        new_markdown_cell(
            "# 03 - Poisson Goals Model\n\n"
            "Fit per-team attack and defense strengths, visualize them, draw a "
            "scoreline heatmap for a fixture, and backtest both models."
        ),
        new_code_cell(
            MINIMAL_SETUP
            + "\nimport numpy as np\n"
            "from wcpredictor.data import load_matches, load_teams\n"
            "from wcpredictor.data.preprocess import build_training_matches\n"
            "from wcpredictor.models import EloModel, PoissonModel\n"
            "from wcpredictor.evaluation import backtest"
        ),
        new_code_cell(
            "tr = build_training_matches(load_matches(config), config)\n"
            "teams = load_teams(config)\n"
            "poisson = PoissonModel(config).fit(tr)\n"
            "strengths = poisson.strengths_table(teams)\n"
            "print('home advantage factor:', round(np.exp(poisson.home_advantage), 3))\n"
            "strengths.head(15)"
        ),
        new_markdown_cell("## Dixon-Coles correction"),
        new_code_cell(
            'print("Fitted Dixon-Coles rho:", round(poisson.rho, 4))'
        ),
        new_code_cell(
            "from dataclasses import replace\n"
            "config_indep = config.with_overrides(poisson=replace(config.poisson, "
            "dixon_coles=False))\n"
            "poisson_indep = PoissonModel(config_indep).fit(tr)\n"
            "m_dc = poisson.scoreline_matrix('France', 'Morocco', True)\n"
            "m_indep = poisson_indep.scoreline_matrix('France', 'Morocco', True)\n"
            "print(f'P(0-0): Dixon-Coles {m_dc[0,0]:.4f}  vs  independent "
            "{m_indep[0,0]:.4f}')\n"
            "print(f'P(1-1): Dixon-Coles {m_dc[1,1]:.4f}  vs  independent "
            "{m_indep[1,1]:.4f}')"
        ),
        new_markdown_cell(
            "With a fitted ρ < 0 (as here), the Dixon-Coles correction pulls "
            "probability mass *toward* the low-scoring draws 0-0 and 1-1 and "
            "*away* from the 1-0/0-1 scorelines, relative to the independent "
            "Poisson model — the classic finding that plain independent "
            "Poisson underestimates low-scoring draws."
        ),
        new_markdown_cell("## Attack vs defense"),
        new_code_cell(
            "fig, ax = plt.subplots()\n"
            "ax.scatter(strengths['attack'], strengths['defense'], alpha=0.6)\n"
            "for _, r in strengths.head(8).iterrows():\n"
            "    ax.annotate(r['team'], (r['attack'], r['defense']))\n"
            "ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)\n"
            "ax.set_xlabel('attack strength'); ax.set_ylabel('defense strength')\n"
            "ax.set_title('Team attack vs defense'); plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Scoreline probabilities for a fixture\n\n"
            "The annotated heatmap tints each scoreline by who it favours "
            "(blue = home win, grey = draw, red = away win) and rings the most "
            "likely score in gold."
        ),
        new_code_cell(
            "from wcpredictor.visualization import (plot_scoreline_heatmap,\n"
            "                                       plot_match_comparison)\n"
            "plot_scoreline_heatmap(poisson, 'France', 'Morocco', neutral=True)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Match outcome: Elo vs Poisson\n\n"
            "Compare the win/draw/loss probabilities the two models assign to "
            "the same fixture."
        ),
        new_code_cell(
            "elo = EloModel(config).fit(tr)\n"
            "home, away = 'France', 'Morocco'\n"
            "preds = {'Elo': elo.predict_match(home, away, True),\n"
            "         'Poisson': poisson.predict_match(home, away, True)}\n"
            "plot_match_comparison(preds, home, away)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Backtest: Elo vs Poisson\n\n"
            "Walk-forward evaluation. Lower log-loss / RPS is better."
        ),
        new_code_cell(
            "from wcpredictor.evaluation import backtest, DEFAULT_MIN_TRAIN, "
            "calibration_table\n"
            "elo_res = backtest(lambda: EloModel(config), tr, config, "
            "min_train=DEFAULT_MIN_TRAIN, collect_predictions=True)\n"
            "poi_res = backtest(lambda: PoissonModel(config), tr, config, "
            "min_train=DEFAULT_MIN_TRAIN, collect_predictions=True)\n"
            "pd.DataFrame([dict(model='Elo', **elo_res.as_dict()),\n"
            "              dict(model='Poisson', **poi_res.as_dict())])"
        ),
        new_markdown_cell(
            "## Calibration\n\n"
            "Reliability diagram: a well-calibrated model's points sit on the "
            "diagonal."
        ),
        new_code_cell(
            "calib = calibration_table(poi_res.predictions)\n"
            "from wcpredictor.visualization import plot_calibration\n"
            "plot_calibration(calib)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell("## Model comparison"),
        new_code_cell(
            "from wcpredictor.visualization import plot_model_comparison\n"
            "plot_model_comparison({'Elo': elo_res, 'Poisson': poi_res})\n"
            "plt.tight_layout(); plt.show()"
        ),
    ]


def notebook_04():
    return [
        new_markdown_cell(
            "# 04 - Tournament Simulation & Outcome Visualizations\n\n"
            "Monte Carlo simulate the 48-team 2026 World Cup, then explore the "
            "results with a set of advanced outcome visualizations from "
            "`wcpredictor.visualization`.\n\n"
            "**The tournament is already under way, so we condition on the "
            "matches that have actually been played.** The group stage and the "
            "completed knockout rounds are settled facts, not things to "
            "re-simulate: `load_tournament_state` reads them from the vendored "
            "2026 fixtures, and `sim.run_conditioned` rolls dice only from the "
            "current bracket onward. A team that has already been eliminated "
            "therefore has a 0% chance of winning -- re-simulating the whole "
            "event from scratch (`sim.run`) would instead hand already-out "
            "sides like Germany or the Netherlands a championship probability "
            "they can no longer have."
        ),
        new_code_cell(
            MINIMAL_SETUP
            + "\nfrom wcpredictor.data import (load_matches, load_groups,\n"
            "                            load_tournament_state)\n"
            "from wcpredictor.data.preprocess import build_training_matches\n"
            "from wcpredictor.models import PoissonModel\n"
            "from wcpredictor.simulation import TournamentSimulator\n"
            "from wcpredictor import visualization as viz\n"
            "import numpy as np"
        ),
        new_code_cell(
            "tr = build_training_matches(load_matches(config), config)\n"
            "poisson = PoissonModel(config).fit(tr)\n"
            "groups = load_groups(config)\n"
            "sim = TournamentSimulator(poisson, groups, config)\n"
            "\n"
            "# Condition on what has actually happened so far. `state` records\n"
            "# how far each team has already gone and the current (frontier)\n"
            "# knockout round; only that round onward is simulated.\n"
            "state = load_tournament_state(config)\n"
            "alive = ', '.join(state.alive)\n"
            "print(f'Current round: {state.frontier_stage.replace(\"_\", \" \")}')\n"
            "print(f'Still alive ({len(state.alive)}): {alive}')\n"
            "report = sim.run_conditioned(\n"
            "    state, n_simulations=config.simulation.n_simulations)\n"
            "report.table.head(12)"
        ),
        new_markdown_cell(
            "### Monte Carlo uncertainty\n\n"
            "Each probability comes from a finite number of simulated "
            "tournaments, so it carries sampling error: "
            "s.e. = sqrt(p(1-p)/n)."
        ),
        new_code_cell("report.standard_errors().head(12)"),
        new_markdown_cell(
            "## The title race\n\n"
            "A lollipop chart of each contender's probability of lifting the "
            "trophy. The horizontal error bars are 95% confidence intervals "
            "(±1.96 s.e.) reflecting Monte Carlo sampling uncertainty, not "
            "additional model uncertainty."
        ),
        new_code_cell(
            "viz.plot_title_race(report, top_n=12)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Full finish distribution\n\n"
            "The headline outcome chart: every bar spans 0-100% and is split "
            "into the *mutually exclusive* ways a team's tournament can end, "
            "from a group-stage exit (left) to champion (right, gold). This "
            "shows a team's entire range of plausible results at once."
        ),
        new_code_cell(
            "viz.plot_outcome_distribution(report, top_n=16)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Reaching each stage (heatmap)\n\n"
            "Cumulative probability of reaching each round."
        ),
        new_code_cell(
            "viz.plot_stage_heatmap(report, top_n=20)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Group-stage outcomes\n\n"
            "Probability of each team finishing 1st / 2nd / 3rd / 4th in their "
            "group (only the top two are guaranteed to advance; the best eight "
            "third-placed teams also progress)."
        ),
        new_code_cell(
            "viz.plot_group_grid(report)\n"
            "plt.show()"
        ),
        new_markdown_cell("## Zoom into a single group"),
        new_code_cell(
            "first_group = sorted(report.table['group'].unique())[0]\n"
            "viz.plot_group_outcomes(report, first_group)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Group difficulty\n\n"
            "Strength-of-schedule view: mean Elo rating per group (hardest "
            "first), with individual team ratings overlaid so a strong "
            "outlier in an otherwise weak group is still visible."
        ),
        new_code_cell(
            "from wcpredictor.models import EloModel\n"
            "from wcpredictor.visualization import plot_group_difficulty\n"
            "elo = EloModel(config).fit(tr)\n"
            "plot_group_difficulty(groups, elo)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## One simulated tournament (example bracket)\n\n"
            "A single random realization of the *remaining* event, rolled "
            "forward from the current bracket. The finalists and semifinalists "
            "are therefore always drawn from the teams still alive."
        ),
        new_code_cell(
            "rng = np.random.default_rng(7)\n"
            "result = sim.simulate_once_conditioned(state, rng)\n"
            "print('Champion:    ', result['champion'])\n"
            "print('Finalists:   ', sorted(result['reached']['final']))\n"
            "print('Semifinalists:', sorted(result['reached']['semifinal']))"
        ),
        new_markdown_cell(
            "## Most likely finish per team\n\n"
            "The single most probable outcome for each of the favourites."
        ),
        new_code_cell(
            "from wcpredictor.simulation.tournament import EXACT_OUTCOMES\n"
            "label_map = dict(EXACT_OUTCOMES)\n"
            "keys = [k for k, _ in EXACT_OUTCOMES]\n"
            "dist = report.outcome_distribution()\n"
            "dist['most_likely'] = dist[keys].idxmax(axis=1).map(label_map)\n"
            "dist['p_most_likely'] = dist[keys].max(axis=1)\n"
            "dist.sort_values('champion', ascending=False)"
            "[['team', 'group', 'most_likely', 'p_most_likely', 'champion']].head(12)"
        ),
    ]


def main():
    # Parse args first so `--help` (or a stray flag) exits cleanly instead of
    # falling through and OVERWRITING every notebook under notebooks/.
    argparse.ArgumentParser(
        description=(
            "Regenerate the analysis notebooks under notebooks/ from this "
            "builder. This OVERWRITES notebooks/0*.ipynb (they are generated "
            "artifacts -- edit this script, not the .ipynb files)."
        )
    ).parse_args()

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    build("01_data_exploration.ipynb", notebook_01())
    build("02_elo_ratings.ipynb", notebook_02())
    build("03_poisson_model.ipynb", notebook_03())
    build("04_tournament_simulation.ipynb", notebook_04())
    print("done")


if __name__ == "__main__":
    main()
