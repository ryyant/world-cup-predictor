"""Generate the analysis notebooks under ``notebooks/``.

Building them programmatically with ``nbformat`` keeps the JSON valid and makes
the notebook contents easy to review and regenerate. Run:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

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


SETUP = """\
import matplotlib.pyplot as plt
import pandas as pd

from wcpredictor.config import default_config
from wcpredictor.data import load_matches, load_teams, load_groups
from wcpredictor.data.preprocess import build_training_matches, team_match_counts

config = default_config()
plt.rcParams["figure.figsize"] = (9, 4.5)
"""


def notebook_01():
    return [
        new_markdown_cell(
            "# 01 - Data Exploration\n\n"
            "Inspect the bundled seed data (historical international results, "
            "team metadata, and the 2026 group draw). The match results are "
            "synthetic but follow a realistic strength hierarchy so the models "
            "have something meaningful to learn. Swap in a real dataset to get "
            "real-world predictions."
        ),
        new_code_cell(SETUP),
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
        new_markdown_cell("## Match outcomes (home / draw / away)"),
        new_code_cell(
            "tr = build_training_matches(matches, config)\n"
            "counts = tr['result'].value_counts().reindex(['H', 'D', 'A'])\n"
            "ax = counts.plot.bar(color=['#4C72B0', '#999999', '#C44E52'])\n"
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
        new_code_cell(SETUP + "\nfrom wcpredictor.models import EloModel"),
        new_code_cell(
            "tr = build_training_matches(load_matches(config), config)\n"
            "teams = load_teams(config)\n"
            "elo = EloModel(config).fit(tr)\n"
            "table = elo.ratings_table(teams)\n"
            "table.head(20)"
        ),
        new_markdown_cell("## Top 15 teams by Elo"),
        new_code_cell(
            "top = table.head(15).iloc[::-1]\n"
            "ax = top.plot.barh(x='team', y='rating', legend=False, "
            "color='#4C72B0')\n"
            "ax.set_title('Top 15 Elo ratings'); ax.set_xlabel('rating')\n"
            "ax.set_xlim(table['rating'].min() - 20, table['rating'].max() + 20)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell("## Average rating by confederation"),
        new_code_cell(
            "by_conf = table.groupby('confederation')['rating']"
            ".mean().sort_values(ascending=False)\n"
            "ax = by_conf.plot.bar(color='#55A868')\n"
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
    ]


def notebook_03():
    return [
        new_markdown_cell(
            "# 03 - Poisson Goals Model\n\n"
            "Fit per-team attack and defense strengths, visualize them, draw a "
            "scoreline heatmap for a fixture, and backtest both models."
        ),
        new_code_cell(
            SETUP
            + "\nimport numpy as np\n"
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
            "elo_res = backtest(lambda: EloModel(config), tr, config, min_train=300)\n"
            "poi_res = backtest(lambda: PoissonModel(config), tr, config, min_train=300)\n"
            "pd.DataFrame([dict(model='Elo', **elo_res.as_dict()),\n"
            "              dict(model='Poisson', **poi_res.as_dict())])"
        ),
    ]


def notebook_04():
    return [
        new_markdown_cell(
            "# 04 - Tournament Simulation & Outcome Visualizations\n\n"
            "Monte Carlo simulate the full 48-team 2026 World Cup, then explore "
            "the results with a set of advanced outcome visualizations from "
            "`wcpredictor.visualization`."
        ),
        new_code_cell(
            SETUP
            + "\nfrom wcpredictor.models import PoissonModel\n"
            "from wcpredictor.simulation import TournamentSimulator\n"
            "from wcpredictor import visualization as viz\n"
            "import numpy as np"
        ),
        new_code_cell(
            "tr = build_training_matches(load_matches(config), config)\n"
            "poisson = PoissonModel(config).fit(tr)\n"
            "groups = load_groups(config)\n"
            "sim = TournamentSimulator(poisson, groups, config)\n"
            "report = sim.run(n_simulations=5000)\n"
            "report.table.head(12)"
        ),
        new_markdown_cell(
            "## The title race\n\n"
            "A lollipop chart of each contender's probability of lifting the "
            "trophy."
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
            "## One simulated tournament (example bracket)\n\n"
            "A single random realization of the whole event."
        ),
        new_code_cell(
            "rng = np.random.default_rng(7)\n"
            "result = sim.simulate_once(rng)\n"
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
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    build("01_data_exploration.ipynb", notebook_01())
    build("02_elo_ratings.ipynb", notebook_02())
    build("03_poisson_model.ipynb", notebook_03())
    build("04_tournament_simulation.ipynb", notebook_04())
    print("done")


if __name__ == "__main__":
    main()
