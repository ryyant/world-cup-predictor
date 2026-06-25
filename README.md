# World Cup 2026 Predictor

A modular Python project that predicts the 48-team 2026 FIFA World Cup using
interpretable statistical models and Monte Carlo simulation, driven by a CLI and
explored through Jupyter notebooks.

- Elo ratings for team strength and win/draw/loss probabilities.
- A Poisson goals model for full scoreline distributions (expected goals,
  goal-difference tiebreakers, knockout extra time / penalties).
- A Monte Carlo simulator of the real 2026 format (12 groups of 4, best
  third-placed teams, a 32-team knockout bracket) that produces per-team
  title and advancement probabilities.

The repo ships with **real World Cup data** so everything runs end to end out of
the box. Match results come from
[openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) and
cover every World Cup finals from 1930 through the already-played 2026 group
games, and the bundled group stage is the **actual 2026 draw**.

> **Coverage note:** the source only includes World Cup *finals* (no qualifiers
> or friendlies), so a few 2026 debutants with no prior finals history (Cape
> Verde, Curaçao, Jordan, Uzbekistan) fall back to the models' default ratings.
> To fold in more matches, see [Using your own data](#using-your-own-data).

## Project layout

```
world-cup-predictor/
├── data/
│   ├── raw/                  bundled CSVs (matches, teams, groups)
│   ├── source/               vendored openfootball worldcup.json (per year)
│   └── processed/            saved model artifacts (generated)
├── notebooks/                01-04 analysis notebooks
├── scripts/
│   ├── fetch_worldcup_data.py build raw CSVs from openfootball data
│   ├── generate_seed_data.py  regenerate synthetic CSVs (offline fallback)
│   └── build_notebooks.py     regenerate the notebooks
├── src/wcpredictor/
│   ├── config.py             paths + model hyperparameters
│   ├── data/                 loader.py, preprocess.py
│   ├── models/               base.py, elo.py, poisson.py
│   ├── simulation/           match.py, tournament.py
│   ├── evaluation/           metrics.py (backtesting)
│   ├── visualization.py      advanced outcome/probability plots
│   └── cli.py                `wcpredict` command-line interface
└── tests/                    pytest suite
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # installs the package + jupyter/pytest
```

(Or `pip install -r requirements.txt` then `pip install -e .`.)

## Quick start (CLI)

```bash
wcpredict train                       # fit Elo + Poisson, save to data/processed
wcpredict ratings --top 20            # current Elo ranking
wcpredict match Brazil Argentina      # predict a single fixture (neutral venue)
wcpredict match France "Saudi Arabia" --home   # TEAM_A at home
wcpredict simulate --n 10000 --top 20 # Monte Carlo the whole tournament
wcpredict simulate --n 10000 --plot outcomes.png  # + save an outcome chart
wcpredict backtest                    # historical accuracy of both models
```

Example simulation output:

```
  #  Team             Grp     Win   Final    Semi   Last8     Adv
  1  France             I   16.7%   25.4%   35.2%   52.1%   94.6%
  2  Croatia            L   10.1%   16.8%   27.4%   42.6%   90.1%
  3  Brazil             H    9.1%   16.8%   26.0%   41.2%   94.2%
  ...
```

## Notebooks

```bash
jupyter lab notebooks/
```

- `01_data_exploration.ipynb` - inspect the seed data and outcome distributions.
- `02_elo_ratings.ipynb` - build and visualize Elo ratings.
- `03_poisson_model.ipynb` - attack/defense strengths, scoreline heatmaps, backtest.
- `04_tournament_simulation.ipynb` - run simulations and the advanced outcome plots.

## Visualizations

`wcpredictor.visualization` turns models and a `SimulationReport` into
publication-quality matplotlib figures. Every function takes an optional `ax`
(or makes its own figure) and returns the `Axes`, so they compose in notebooks.

```python
from wcpredictor import visualization as viz

viz.plot_outcome_distribution(report)  # stacked finish distribution per team
viz.plot_stage_heatmap(report)         # reach-stage probabilities (heatmap)
viz.plot_title_race(report)            # championship-odds lollipop chart
viz.plot_group_grid(report)            # group finishing positions (1st-4th)
viz.plot_group_outcomes(report, "A")   # one group in detail
viz.plot_scoreline_heatmap(poisson, "France", "Morocco")   # W/D/L-tinted grid
viz.plot_match_comparison(preds, "France", "Morocco")      # model comparison
```

The headline chart, `plot_outcome_distribution`, shows each team's *full* range
of outcomes in mutually exclusive segments (group-stage exit -> champion), so a
single bar communicates how far a team is likely to go. These finish
probabilities are derived from the cumulative reach-stage probabilities via
`report.outcome_distribution()`; group-position probabilities are available via
`report.group_position_distribution()`. The CLI's `simulate --plot PATH` saves
the outcome-distribution chart directly.

## Methodology

### Elo
Ratings start at 1500 and update after each match (replayed chronologically):

```
expected_home = 1 / (1 + 10^((R_away - R_home - home_adv) / 400))
R_home += K * MOV * (actual - expected_home)
```

A margin-of-victory multiplier (`MOV`) dampens blowouts. The continuous expected
score is split into win/draw/loss probabilities with a simple draw model whose
draw probability peaks at an even matchup and vanishes for lopsided fixtures.

### Poisson
Per-team attack and defense strengths are fit by weighted Poisson maximum
likelihood (recency-weighted, with ridge shrinkage toward the mean):

```
log E[goals] = intercept + home_adv * is_home + attack[scorer] - defense[opponent]
```

Goals for the two teams are treated as independent Poisson variables, giving a
full scoreline distribution. This drives goal-difference tiebreakers and
realistic knockout results (extra time as a scaled-down Poisson, then a
penalty-shootout lottery weighted slightly toward the stronger side).

### Tournament simulation
Each Monte Carlo run plays all 12 groups (with configurable tiebreakers:
points, goal difference, goals for, head-to-head, then a random drawing of
lots), ranks the third-placed teams to pick the best 8, seeds the 32 qualifiers
into a balanced single-elimination bracket, and plays it out. Host nations
(USA, Mexico, Canada) receive home advantage. Aggregating thousands of runs
yields each team's probability of advancing and reaching each stage.

> Note: the knockout bracket uses a seeding-based pairing (a deterministic,
> balanced approximation) rather than FIFA's exact third-place slotting table,
> which depends on which groups the qualifying third-placed teams come from.

### Evaluation
`wcpredict backtest` runs a walk-forward backtest reporting accuracy, multiclass
log-loss, and the ranked probability score (RPS, the standard metric for ordered
football outcomes; lower is better).

## Configuration

All hyperparameters live in [`src/wcpredictor/config.py`](src/wcpredictor/config.py)
(Elo K-factor and home advantage, Poisson recency half-life and shrinkage,
number of simulations, tiebreaker order, random seed). Build a custom `Config`
in a notebook to experiment without touching the rest of the code.

## Using your own data

The bundled data covers World Cup finals only. For richer predictions you can
swap in a broader dataset (qualifiers, friendlies, etc.). The models only need a
`matches.csv` with these columns:

| column      | description                              |
|-------------|------------------------------------------|
| `date`      | match date (any parseable format)        |
| `home_team` | home team name                           |
| `away_team` | away team name                           |
| `home_score`| goals scored by the home team (int)      |
| `away_score`| goals scored by the away team (int)      |
| `neutral`   | 1 if played at a neutral venue, else 0   |
| `tournament`| competition name (free text)             |

To use real data:

1. Replace `data/raw/matches.csv` with your dataset (e.g. a Kaggle
   "International football results" export). Keep the column names above.
2. Update `data/raw/teams.csv` (`team`, `confederation`, `pot`) and
   `data/raw/wc2026_groups.csv` (`group`, `slot`, `team`) so every group has 4
   teams and team names match `matches.csv`.
3. Re-run `wcpredict train` and `wcpredict simulate`.

To regenerate the bundled data (re-reads the vendored `data/source/*.json`) or
the notebooks:

```bash
python scripts/fetch_worldcup_data.py            # real World Cup data (default)
python scripts/fetch_worldcup_data.py --refresh  # re-download source JSON first
python scripts/generate_seed_data.py             # synthetic data (offline fallback)
python scripts/build_notebooks.py                # rebuild the notebooks
```

A live football API could be wired in behind `data/loader.py` by writing a
fetcher that produces the same `matches.csv` schema.

## Development

```bash
pytest          # run the test suite
```

## License

MIT
