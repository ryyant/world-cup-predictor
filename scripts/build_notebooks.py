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


# --------------------------------------------------------------------------
# Per-phase prediction notebooks (04-08).
#
# One notebook per tournament phase. Each freezes the event at the start of a
# round, trains the model on ONLY the matches available at that point (strict
# point-in-time, no look-ahead), predicts that round and everything after via a
# conditioned simulation, and -- for rounds that have since been played --
# scores those predictions against the real results. They share the frontier
# machinery in wcpredictor.data.tournament_state (as_of_stage / phase_start_dates
# / actual_knockout_ties).
# --------------------------------------------------------------------------

# Team-A / team-B tint for the knockout "who advances" bars, kept consistent
# with the home/away language elsewhere in the notebooks.
TIE_A_COLOR = NB_PALETTE["home"]
TIE_B_COLOR = NB_PALETTE["away"]


def notebook_group_stage():
    return [
        new_markdown_cell(
            "# 04 - Phase 1: Group Stage (pre-tournament)\n\n"
            "The first of five phase notebooks. Each one freezes the tournament "
            "at the start of a round, trains a model on **only the matches "
            "available at that point** (strict point-in-time -- no look-ahead), "
            "and predicts what happens next; then, for rounds that have since "
            "been played, it scores those predictions against reality.\n\n"
            "**This is the pre-tournament view.** The model is trained on every "
            "match played *before the 2026 group stage kicked off*, so it has "
            "never seen a single 2026 result. We predict (1) the win/draw/loss "
            "odds of each group fixture and (2) every team's chance of advancing "
            "to the knockout -- then check both against what actually happened."
        ),
        new_code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "from wcpredictor.config import default_config\n"
            "from wcpredictor.data import (load_matches, load_groups,\n"
            "                              load_tournament_state,\n"
            "                              build_training_matches, phase_start_dates)\n"
            "from wcpredictor.data.tournament_state import GROUP_STAGE\n"
            "from wcpredictor.models import PoissonModel\n"
            "from wcpredictor.simulation import TournamentSimulator\n"
            "from wcpredictor import visualization as viz\n\n"
            "config = default_config()\n"
            "plt.rcParams['figure.figsize'] = (9, 4.5)"
        ),
        new_markdown_cell(
            "## Point-in-time training\n\n"
            "Cut the data at the first 2026 group match: the model sees only "
            "matches played strictly before it."
        ),
        new_code_cell(
            "matches = load_matches(config)\n"
            "dates = phase_start_dates(config)\n"
            "cutoff = pd.Timestamp(dates[GROUP_STAGE])   # first 2026 group match\n"
            "train = matches[matches['date'] < cutoff]\n"
            "print(f'Group stage starts {cutoff.date()}.')\n"
            "print(f'Training on {len(train)} pre-tournament matches; '\n"
            "      f'holding out {len(matches) - len(train)} played since.')\n"
            "tr = build_training_matches(train, config, reference_date=cutoff)\n"
            "poisson = PoissonModel(config).fit(tr)"
        ),
        new_markdown_cell(
            "## Match-by-match group predictions\n\n"
            "For every actual group fixture: the model's win / draw / loss "
            "probabilities (home perspective), the outcome it called most "
            "likely, and what really happened. We zoom into one group here; the "
            "accuracy summary that follows covers all 72 group matches."
        ),
        new_code_cell(
            "r32_start = pd.Timestamp(dates['round_of_32'])\n"
            "group_games = matches[(matches['tournament'] == 'World Cup 2026') &\n"
            "                      (matches['date'] >= cutoff) &\n"
            "                      (matches['date'] < r32_start)].copy()\n\n"
            "def _predict_row(g):\n"
            "    pred = poisson.predict_match(g['home_team'], g['away_team'],\n"
            "                                 neutral=bool(g['neutral']))\n"
            "    actual = ('H' if g['home_score'] > g['away_score']\n"
            "              else 'A' if g['away_score'] > g['home_score'] else 'D')\n"
            "    return pd.Series({\n"
            "        'home': g['home_team'], 'away': g['away_team'],\n"
            "        'p_home': round(pred.p_home_win, 3),\n"
            "        'p_draw': round(pred.p_draw, 3),\n"
            "        'p_away': round(pred.p_away_win, 3),\n"
            "        'called': pred.most_likely,\n"
            "        'score': f\"{g['home_score']}-{g['away_score']}\",\n"
            "        'actual': actual, 'hit': pred.most_likely == actual})\n\n"
            "preds = group_games.apply(_predict_row, axis=1)\n"
            "example = sorted(load_groups(config)['group'].unique())[0]\n"
            "in_group = set(load_groups(config).query('group == @example')['team'])\n"
            "print(f'Group {example} fixtures:')\n"
            "preds[preds['home'].isin(in_group) & preds['away'].isin(in_group)]"
        ),
        new_markdown_cell(
            "## How good were the match predictions?\n\n"
            "Scored across all 72 group games with accuracy, multiclass "
            "log-loss, and the ranked probability score (RPS) -- the standard "
            "metric for ordered football outcomes. Lower log-loss / RPS is "
            "better."
        ),
        new_code_cell(
            "def _rps(row):\n"
            "    order = ['H', 'D', 'A']\n"
            "    p = np.array([row['p_home'], row['p_draw'], row['p_away']])\n"
            "    y = np.eye(3)[order.index(row['actual'])]\n"
            "    return np.sum((np.cumsum(p) - np.cumsum(y)) ** 2) / 2\n\n"
            "def _p_actual(row):\n"
            "    return {'H': row['p_home'], 'D': row['p_draw'],\n"
            "            'A': row['p_away']}[row['actual']]\n\n"
            "acc = preds['hit'].mean()\n"
            "ll = -np.log(np.clip(preds.apply(_p_actual, axis=1), 1e-15, 1)).mean()\n"
            "rps = preds.apply(_rps, axis=1).mean()\n"
            "print(f'{len(preds)} group matches | accuracy {acc:.0%} | '\n"
            "      f'log-loss {ll:.3f} | RPS {rps:.3f}')\n"
            "print('(A three-way coin flip scores log-loss 1.099 / RPS ~0.22.)')"
        ),
        new_markdown_cell(
            "## Who progresses? (Monte Carlo, from scratch)\n\n"
            "The odds above are single-game views. To turn them into "
            "*advancement* probabilities we simulate the whole group stage (and "
            "the knockout beyond) many times with `sim.run()` -- the "
            "pre-tournament projection in which all 48 teams still contend. Each "
            "team's finishing-position and advance probabilities fall out of the "
            "aggregate."
        ),
        new_code_cell(
            "sim = TournamentSimulator(poisson, load_groups(config), config)\n"
            "report = sim.run(n_simulations=config.simulation.n_simulations)\n"
            "prog = report.table[['team', 'group', 'p_group_1st', 'p_group_2nd',\n"
            "                     'p_group_3rd', 'p_group_4th', 'p_advance']]\n"
            "prog.sort_values(['group', 'p_advance'],\n"
            "                 ascending=[True, False]).head(12)"
        ),
        new_code_cell(
            "viz.plot_group_grid(report)\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## Did the right teams progress?\n\n"
            "The 32 teams that actually reached the knockout come straight from "
            "the played results (`load_tournament_state(...).reached`). We "
            "compare each team's simulated advance probability with whether it "
            "truly went through."
        ),
        new_code_cell(
            "actual_adv = set(load_tournament_state(config).reached)  # reached R32+\n"
            "adv = report.table[['team', 'p_advance']].copy()\n"
            "adv['advanced'] = adv['team'].isin(actual_adv).astype(float)\n"
            "p = adv['p_advance'].clip(1e-15, 1 - 1e-15)\n"
            "brier = np.mean((adv['p_advance'] - adv['advanced']) ** 2)\n"
            "ll = -np.mean(adv['advanced'] * np.log(p) +\n"
            "              (1 - adv['advanced']) * np.log(1 - p))\n"
            "top = set(adv.sort_values('p_advance', ascending=False)\n"
            "          .head(len(actual_adv))['team'])\n"
            "print(f'{len(adv)} teams | Brier {brier:.3f} | log-loss {ll:.3f}')\n"
            "print(f'Of the {len(actual_adv)} teams we ranked most likely to "
            "advance, {len(top & actual_adv)} actually did.')\n"
            "adv['surprise'] = adv['advanced'] - adv['p_advance']\n"
            "cols = ['team', 'p_advance', 'advanced']\n"
            "print('\\nBiggest upsets (advanced despite low odds):')\n"
            "display(adv.sort_values('surprise', ascending=False).head(5)[cols])\n"
            "print('Biggest flops (favoured but eliminated):')\n"
            "display(adv.sort_values('surprise').head(5)[cols])"
        ),
        new_markdown_cell(
            "## Pre-tournament title odds\n\n"
            "For completeness, the from-scratch championship picture *before a "
            "ball was kicked*."
        ),
        new_code_cell(
            "viz.plot_title_race(report, top_n=12)\n"
            "plt.tight_layout(); plt.show()"
        ),
    ]


def _knockout_phase_notebook(number, phase, title, stage_key, played):
    """Cells for a knockout-phase notebook (R32 / R16 / QF / SF).

    ``played`` toggles the scorecard narration between "grade it" (a round that
    has happened) and "live prediction" (a round still to come).
    """
    if played:
        head_note = (
            "Because this round has since been played, we also **score our "
            "predictions against the actual results** at the end."
        )
        score_note = (
            "Now that the round has been played, we grade every tie prediction."
        )
    else:
        head_note = (
            "This round has **not been played yet** in the vendored data, so "
            "this is a live prediction -- there is nothing to grade against yet."
        )
        score_note = (
            "This round hasn't been played in the vendored data yet, so the "
            "cell just lists the live predictions -- rerun it once the results "
            "are in to grade them."
        )
    return [
        new_markdown_cell(
            f"# {number} - Phase {phase}: {title}\n\n"
            f"Freezes the tournament at the start of the {title.lower()} and "
            "predicts it. The model is trained only on matches played **before "
            "this round began**, and the simulation is *conditioned* on "
            "everything already settled: the group stage and every earlier "
            "knockout round are locked in, and we roll dice only from this round "
            f"onward. {head_note}"
        ),
        new_code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "from wcpredictor.config import default_config\n"
            "from wcpredictor.data import (load_matches, load_groups,\n"
            "                              load_tournament_state,\n"
            "                              build_training_matches, phase_start_dates,\n"
            "                              actual_knockout_ties)\n"
            "from wcpredictor.models import PoissonModel\n"
            "from wcpredictor.simulation import TournamentSimulator\n"
            "from wcpredictor.simulation.tournament import STAGES, _STAGE_INDEX\n"
            "from wcpredictor import visualization as viz\n\n"
            "config = default_config()\n"
            "plt.rcParams['figure.figsize'] = (9, 4.5)\n"
            f"STAGE = {stage_key!r}\n"
            "NEXT = STAGES[_STAGE_INDEX[STAGE] + 1]  # the round winners advance to\n"
            f"A_COLOR, B_COLOR = {TIE_A_COLOR!r}, {TIE_B_COLOR!r}\n"
            "label = STAGE.replace('_', ' ')"
        ),
        new_markdown_cell(
            "## Point-in-time training\n\n"
            "Train on every match played strictly before this round started, so "
            "the model never sees a result it is about to predict."
        ),
        new_code_cell(
            "matches = load_matches(config)\n"
            "cutoff = pd.Timestamp(phase_start_dates(config)[STAGE])\n"
            "train = matches[matches['date'] < cutoff]\n"
            "print(f'The {label} starts {cutoff.date()}.')\n"
            "print(f'Training on {len(train)} matches played before it; '\n"
            "      f'holding out {len(matches) - len(train)} later matches.')\n"
            "tr = build_training_matches(train, config, reference_date=cutoff)\n"
            "poisson = PoissonModel(config).fit(tr)"
        ),
        new_markdown_cell(
            "## Condition on everything settled, then simulate forward\n\n"
            "`as_of_stage` rewinds the fixtures to the start of this round; "
            "`run_conditioned` locks in the settled prefix and rolls the rest "
            "of the bracket forward."
        ),
        new_code_cell(
            "state = load_tournament_state(config, as_of_stage=STAGE)\n"
            "sim = TournamentSimulator(poisson, load_groups(config), config)\n"
            "report = sim.run_conditioned(\n"
            "    state, n_simulations=config.simulation.n_simulations)\n"
            "print(f'{len(state.alive)} teams enter the {label}:')\n"
            "print('  ' + ', '.join(state.alive))\n"
            "report.table[report.table['team'].isin(state.alive)]"
        ),
        new_markdown_cell(
            "## This round's ties: who advances?\n\n"
            "In a knockout tie a team's probability of *reaching the next round* "
            "is exactly its probability of winning the tie -- and it already "
            "folds in extra time and penalties. The two sides of each tie sum "
            "to 100%."
        ),
        new_code_cell(
            "probs = report.table.set_index('team')[f'p_{NEXT}']\n"
            "rows = []\n"
            "for i, tie in enumerate(state.frontier, 1):\n"
            "    p1 = float(probs[tie.team1])\n"
            "    rows.append({'#': i, 'team A': tie.team1, 'p(A adv)': round(p1, 3),\n"
            "                 'team B': tie.team2, 'p(B adv)': round(1 - p1, 3),\n"
            "                 'favourite': tie.team1 if p1 >= 0.5 else tie.team2})\n"
            "pd.DataFrame(rows)"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(9, 0.6 * len(state.frontier) + 1))\n"
            "for i, tie in enumerate(state.frontier):\n"
            "    p1 = float(probs[tie.team1])\n"
            "    ax.barh(i, p1, color=A_COLOR)\n"
            "    ax.barh(i, 1 - p1, left=p1, color=B_COLOR)\n"
            "    ax.text(0.01, i, f'{tie.team1}  {p1:.0%}', va='center', ha='left',\n"
            "            color='white', fontsize=9, fontweight='bold')\n"
            "    ax.text(0.99, i, f'{1 - p1:.0%}  {tie.team2}', va='center',\n"
            "            ha='right', color='white', fontsize=9, fontweight='bold')\n"
            "ax.axvline(0.5, color='white', lw=1, ls='--')\n"
            "ax.set_yticks([]); ax.set_xlim(0, 1); ax.invert_yaxis()\n"
            "ax.set_xlabel('probability of advancing')\n"
            "ax.set_title(f'{label.title()}: who advances?')\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## How far do they go?\n\n"
            "Rolling the same conditioned simulation to the final gives each "
            "surviving team's title odds and full finish distribution."
        ),
        new_code_cell(
            "viz.plot_title_race(report, top_n=len(state.alive))\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_code_cell(
            "viz.plot_outcome_distribution(report, top_n=len(state.alive))\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_code_cell(
            "viz.plot_stage_heatmap(report, top_n=len(state.alive))\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Scorecard: predictions vs reality\n\n" + score_note
        ),
        new_code_cell(
            "actual = {frozenset((t.team1, t.team2)): t.winner\n"
            "          for t in actual_knockout_ties(config).get(STAGE, ())}\n"
            "recs, ps, ys = [], [], []\n"
            "for tie in state.frontier:\n"
            "    p1 = float(probs[tie.team1])\n"
            "    w = actual.get(frozenset((tie.team1, tie.team2)))\n"
            "    rec = {'match': f'{tie.team1} v {tie.team2}',\n"
            "           'predicted': tie.team1 if p1 >= 0.5 else tie.team2,\n"
            "           'confidence': round(max(p1, 1 - p1), 3),\n"
            "           'actual': w or 'not played yet'}\n"
            "    if w:\n"
            "        y = 1.0 if w == tie.team1 else 0.0\n"
            "        ps.append(p1); ys.append(y)\n"
            "        rec['result'] = 'correct' if (p1 >= 0.5) == (y == 1.0) else 'upset'\n"
            "    recs.append(rec)\n"
            "scorecard = pd.DataFrame(recs)\n"
            "if ps:\n"
            "    ps, ys = np.array(ps), np.array(ys)\n"
            "    pc = np.clip(ps, 1e-15, 1 - 1e-15)\n"
            "    acc = np.mean((pc >= 0.5) == (ys == 1))\n"
            "    brier = np.mean((ps - ys) ** 2)\n"
            "    ll = -np.mean(ys * np.log(pc) + (1 - ys) * np.log(1 - pc))\n"
            "    print(f'{len(ys)} ties scored | accuracy {acc:.0%} | '\n"
            "          f'Brier {brier:.3f} | log-loss {ll:.3f}')\n"
            "    print('(Lower Brier/log-loss is better; a 50/50 guess scores "
            "0.250 / 0.693.)')\n"
            "else:\n"
            "    print(f'The {label} has not been played yet -- predictions only.')\n"
            "scorecard"
        ),
    ]


def notebook_round_of_32():
    return _knockout_phase_notebook(
        "05", 2, "Round of 32", "round_of_32", played=True)


def notebook_round_of_16():
    return _knockout_phase_notebook(
        "06", 3, "Round of 16", "round_of_16", played=True)


def notebook_quarterfinals():
    return _knockout_phase_notebook(
        "07", 4, "Quarterfinals", "quarterfinal", played=True)


def notebook_semifinals():
    return _knockout_phase_notebook(
        "08", 5, "Semifinals", "semifinal", played=False)


def notebook_knockout_scores():
    """Historical analysis notebook: late-knockout scores, 2006-2022.

    Not part of the per-phase prediction series -- it looks *backward* at how
    quarterfinal-onward matches actually score, as an empirical reference for
    the simulator's knockout machinery (Poisson scorelines -> scaled-down
    extra time -> penalties).
    """
    return [
        new_markdown_cell(
            "# 90 - Knockout scores: quarterfinals onward, 2006-2026\n\n"
            "How do late-knockout matches actually behave? This notebook takes "
            "every match from the quarterfinals on -- quarterfinals, semifinals, "
            "the third-place match and the final -- across the last five "
            "*completed* World Cups (2006-2022, eight such matches each) plus "
            "whatever the in-progress 2026 edition has already played from the "
            "quarterfinals onward, and looks at the distribution of **full-time "
            "(90-minute) scores** and of the scores **after extra time** for "
            "ties that were level.\n\n"
            "These matches are the empirical base rates that the simulator's "
            "knockout machinery (Poisson scorelines, then extra time as a "
            "scaled-down match, then a penalty lottery) is meant to reproduce, "
            "so at the end we check its extra-time scaling assumption against "
            "the data. Unplayed 2026 fixtures are skipped, so the sample grows "
            "on its own as the tournament finishes and the data is refreshed."
        ),
        new_code_cell(
            "import json\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "from wcpredictor.config import default_config\n"
            "from wcpredictor.simulation.match import EXTRA_TIME_FRACTION\n\n"
            "config = default_config()\n"
            "plt.rcParams['figure.figsize'] = (9, 4.5)\n\n"
            "SOURCE_DIR = config.wc2026_source_path.parent\n"
            "YEARS = [2006, 2010, 2014, 2018, 2022, 2026]\n"
            "COMPLETED = [y for y in YEARS if y != 2026]\n"
            "# openfootball's round labels vary by year; canonicalise QF onward.\n"
            "ROUND_LABELS = {\n"
            "    'Quarterfinals': 'QF', 'Quarter-finals': 'QF', 'Quarter-final': 'QF',\n"
            "    'Semifinals': 'SF', 'Semi-finals': 'SF', 'Semi-final': 'SF',\n"
            "    'Third-place play-off': '3rd place',\n"
            "    'Match for third place': '3rd place',\n"
            "    'Final': 'Final',\n"
            "}\n"
            "STAGE_ORDER = ['QF', 'SF', '3rd place', 'Final']"
        ),
        new_markdown_cell(
            "## The matches\n\n"
            "Read straight from the vendored `data/source/*.worldcup.json` "
            "files rather than `matches.csv`, because only the source JSON "
            "keeps the full-time and after-extra-time scores apart: `score.ft` "
            "is the 90-minute score and `score.et`, when present, the "
            "cumulative score after 120 minutes (`score.p` records a "
            "shootout). The *settled* score is the one on the scoreboard when "
            "the tie was decided: after extra time where it was played, after "
            "90 minutes otherwise. Fixtures with no full-time score yet (the "
            "still-to-be-played 2026 rounds) are skipped."
        ),
        new_code_cell(
            "rows = []\n"
            "for year in YEARS:\n"
            "    doc = json.loads((SOURCE_DIR / f'{year}.worldcup.json')\n"
            "                     .read_text(encoding='utf-8'))\n"
            "    matches = list(doc.get('matches', []))\n"
            "    for rnd in doc.get('rounds', []):\n"
            "        matches.extend(rnd.get('matches', []))\n"
            "    for m in matches:\n"
            "        stage = ROUND_LABELS.get((m.get('round') or '').strip())\n"
            "        if stage is None:\n"
            "            continue\n"
            "        s = m.get('score') or {}\n"
            "        ft, et, pens = s.get('ft'), s.get('et'), s.get('p')\n"
            "        if ft is None:\n"
            "            continue  # not played yet (2026 rounds still to come)\n"
            "        fin = et or ft  # the settled score (see above)\n"
            "        rows.append({'year': year, 'stage': stage,\n"
            "                     'team1': m['team1'], 'team2': m['team2'],\n"
            "                     'ft1': ft[0], 'ft2': ft[1],\n"
            "                     'fin1': fin[0], 'fin2': fin[1],\n"
            "                     'extra_time': et is not None,\n"
            "                     'penalties': pens is not None})\n"
            "df = pd.DataFrame(rows)\n"
            "df['stage'] = pd.Categorical(df['stage'], STAGE_ORDER, ordered=True)\n"
            "per_year = df.groupby('year').size()\n"
            "assert (per_year.loc[COMPLETED] == 8).all(), \\\n"
            "    'expected 8 matches per completed WC from the QF on'\n"
            "print(f'{len(df)} matches across {len(YEARS)} World Cups '\n"
            "      f'({per_year.get(2026, 0)} of them from the in-progress 2026):')\n"
            "print(per_year.to_string())\n"
            "df.head()"
        ),
        new_markdown_cell(
            "## How often does 90 minutes fail to settle it?\n\n"
            "In a knockout, a full-time draw *is* extra time, so the FT draw "
            "rate doubles as the extra-time rate."
        ),
        new_code_cell(
            "summary = (df.groupby('stage', observed=True)\n"
            "             .agg(matches=('year', 'size'),\n"
            "                  extra_time=('extra_time', 'sum'),\n"
            "                  penalties=('penalties', 'sum')))\n"
            "summary['ft_draw_rate'] = summary['extra_time'] / summary['matches']\n"
            "print(f\"Overall: {df['extra_time'].mean():.0%} level after 90 minutes \"\n"
            "      f\"({df['extra_time'].sum()} of {len(df)}); \"\n"
            "      f\"{df['penalties'].sum()} decided on penalties.\")\n"
            "summary"
        ),
        new_markdown_cell(
            "## Full-time scorelines\n\n"
            "Scorelines as winner-loser (venue is irrelevant at a neutral "
            "tournament), draws in grey. Late knockouts are tight, low-scoring "
            "affairs -- compare the group-stage goal distributions in notebook "
            "01."
        ),
        new_code_cell(
            "def scoreline(a, b):\n"
            "    return f'{max(a, b)}-{min(a, b)}'\n\n"
            "def by_goals(index):\n"
            "    return sorted(index, key=lambda s: (sum(map(int, s.split('-'))), s))\n\n"
            "ft_scores = df.apply(lambda r: scoreline(r['ft1'], r['ft2']), axis=1)\n"
            "counts = ft_scores.value_counts().loc[by_goals(ft_scores.unique())]\n"
            "is_draw = [s.split('-')[0] == s.split('-')[1] for s in counts.index]\n"
            f"colors = [{NB_PALETTE['draw']!r} if d else {NB_PALETTE['home']!r} "
            "for d in is_draw]\n"
            "ax = counts.plot.bar(color=colors, rot=0)\n"
            "ax.set_title(f'Full-time scorelines, QF onward, '\n"
            "             f'{YEARS[0]}-{YEARS[-1]} (draws in grey)')\n"
            "ax.set_xlabel('scoreline (winner-loser)'); ax.set_ylabel('matches')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"mean FT goals/match: {(df['ft1'] + df['ft2']).mean():.2f} \"\n"
            "      f\"(vs {counts.sum()} matches)\")"
        ),
        new_markdown_cell(
            "## ...and after extra time: the settled scorelines\n\n"
            "The same distribution once the level ties have played their extra "
            "30 minutes. Draw cells shrink but do not vanish -- a scoreline "
            "still level after 120 minutes means the tie went to penalties."
        ),
        new_code_cell(
            "fin_scores = df.apply(lambda r: scoreline(r['fin1'], r['fin2']), axis=1)\n"
            "comp = (pd.DataFrame({'full time': ft_scores.value_counts(),\n"
            "                      'after extra time': fin_scores.value_counts()})\n"
            "        .fillna(0).astype(int))\n"
            "comp = comp.loc[by_goals(comp.index)]\n"
            f"ax = comp.plot.bar(color=[{NB_PALETTE['home']!r}, "
            f"{NB_PALETTE['accent']!r}], rot=0)\n"
            "ax.set_title('Scoreline distribution: 90 minutes vs settled')\n"
            "ax.set_xlabel('scoreline (winner-loser)'); ax.set_ylabel('matches')\n"
            "plt.tight_layout(); plt.show()\n"
            "still_level = fin_scores[df['penalties']].value_counts()\n"
            "print('Settled-score draws (decided on penalties):')\n"
            "print(still_level.to_string())"
        ),
        new_markdown_cell("### Total goals, before and after extra time"),
        new_code_cell(
            "tg = (pd.DataFrame({'full time': (df['ft1'] + df['ft2']).value_counts(),\n"
            "                    'settled': (df['fin1'] + df['fin2']).value_counts()})\n"
            "      .fillna(0).astype(int).sort_index())\n"
            f"ax = tg.plot.bar(color=[{NB_PALETTE['home']!r}, "
            f"{NB_PALETTE['accent']!r}], rot=0)\n"
            "ax.set_title(f'Total goals per match, QF onward '\n"
            "             f'({YEARS[0]}-{YEARS[-1]})')\n"
            "ax.set_xlabel('goals'); ax.set_ylabel('matches')\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Does extra time score like a third of a match?\n\n"
            "The simulator plays extra time as the same Poisson attack/defense "
            "matchup scaled by `EXTRA_TIME_FRACTION` (30/90 of the expected "
            "goals -- see `wcpredictor.simulation.match`). Check that "
            "assumption against these tournaments: goals actually scored in "
            "extra-time periods vs what a third of the observed 90-minute "
            "scoring rate would predict."
        ),
        new_code_cell(
            "et_games = df[df['extra_time']]\n"
            "et_goals = ((et_games['fin1'] - et_games['ft1'])\n"
            "            + (et_games['fin2'] - et_games['ft2']))\n"
            "ft_rate = (df['ft1'] + df['ft2']).mean()\n"
            "print(f'{len(et_games)} matches went to extra time; '\n"
            "      f'{int(et_goals.sum())} goals were scored in those periods.')\n"
            "print(f'observed goals per ET period:            {et_goals.mean():.2f}')\n"
            "print(f'FT rate x EXTRA_TIME_FRACTION ({EXTRA_TIME_FRACTION:.2f}): '\n"
            "      f'{ft_rate * EXTRA_TIME_FRACTION:.2f}')\n"
            "se = et_goals.std(ddof=1) / np.sqrt(len(et_goals))\n"
            "print(f'(+/-1 s.e. on the observed rate: {se:.2f} -- '\n"
            "      f'{len(et_goals)} matches is a small sample)')\n"
            "print(f'ET matches settled without penalties: '\n"
            "      f'{1 - et_games[\"penalties\"].mean():.0%}')"
        ),
        new_markdown_cell(
            "### Reading the result\n\n"
            "Two caveats before over-interpreting any gap between the two "
            "rates. The sample is tiny (a handful of extra-time matches per "
            "tournament), so the standard error above can span the whole "
            "difference. And matches that reach extra time are not a random "
            "sample: they are ties between well-matched sides, played on "
            "tired legs -- caution and fatigue push scoring down, while the "
            "urgency of avoiding a shootout pushes it up, so the selection "
            "effects cut both ways. If the observed rate lands near the "
            "one-third scaling, that is direct empirical support for the "
            "simulator's independent-Poisson extra time (deliberately "
            "without the Dixon-Coles draw correction, which models 90-minute "
            "dependence); a persistent gap in either direction would justify "
            "fitting a separate extra-time intercept from exactly this data."
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
    # Per-phase prediction notebooks (point-in-time training + conditioning),
    # in tournament order.
    build("04_group_stage.ipynb", notebook_group_stage())
    build("05_round_of_32.ipynb", notebook_round_of_32())
    build("06_round_of_16.ipynb", notebook_round_of_16())
    build("07_quarterfinals.ipynb", notebook_quarterfinals())
    build("08_semifinals.ipynb", notebook_semifinals())
    # Reference/appendix analyses live in the 9x range, out of the phase
    # series' way (the next phase notebook will be 09_final.ipynb).
    build("90_knockout_scores.ipynb", notebook_knockout_scores())
    print("done")


if __name__ == "__main__":
    main()
