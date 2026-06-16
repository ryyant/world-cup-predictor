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
> or friendlies), so 2026 debutants with no prior finals history fall back to
> the models' default ratings. As of the currently vendored data that's just
> **Uzbekistan** -- the other 2026 debutants already have a `matches.csv` row
> from the 20 already-played 2026 group games. This list can shrink further (or
> shift) as more 2026 matches get vendored, so don't rely on a hardcoded name
> here: `notebooks/01_data_exploration.ipynb` computes it live against
> whatever data is currently bundled. To fold in more matches, see
> [Using your own data](#using-your-own-data).

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
wcpredict simulate --n 10000 --top 20 # Monte Carlo from the current bracket
wcpredict simulate --n 10000 --from-scratch  # ignore results so far (pre-tournament)
wcpredict simulate --n 10000 --plot outcomes.png  # + save an outcome chart
wcpredict backtest                    # historical accuracy of both models
```

By default `simulate` **conditions on the matches already played** (read from
the vendored 2026 fixtures): the group stage and completed knockout rounds are
locked in, and only the current bracket onward is rolled forward, so
already-eliminated teams correctly show a 0% title chance. Example output
mid-tournament (`wcpredict simulate --top 10`, at the quarterfinal stage):

```
Conditioning on results through the quarterfinal: 8 teams still alive.

  #  Team             Grp     Win   Final    Semi   Last8     Adv
  1  France             I   24.2%   36.7%   63.6%  100.0%  100.0%
  2  Spain              H   19.5%   30.9%   60.2%  100.0%  100.0%
  3  England            L   16.1%   36.2%   63.7%  100.0%  100.0%
  4  Argentina          J   15.9%   37.0%   68.5%  100.0%  100.0%
  5  Belgium            G    8.5%   16.5%   39.8%  100.0%  100.0%
  ...
```

Pass `--from-scratch` for the pre-tournament projection that replays the whole
event (all 48 teams contend, ignoring who has since been knocked out).

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
viz.plot_title_race(report)            # championship-odds lollipop chart (+ 95% CI error bars)
viz.plot_group_grid(report)            # group finishing positions (1st-4th)
viz.plot_group_outcomes(report, "A")   # one group in detail
viz.plot_scoreline_heatmap(poisson, "France", "Morocco")   # W/D/L-tinted grid
viz.plot_match_comparison(preds, "France", "Morocco")      # model comparison
viz.plot_calibration(calib)                           # reliability diagram (predicted vs empirical)
viz.plot_rating_history(elo.history, ["Brazil", ...])  # Elo trajectories over time (needs fit(track_history=True))
viz.plot_model_comparison({"Elo": elo_res, "Poisson": poi_res})  # backtest metrics side by side
viz.plot_group_difficulty(groups, elo)                # "group of death" chart by mean Elo
```

The headline chart, `plot_outcome_distribution`, shows each team's *full* range
of outcomes in mutually exclusive segments (group-stage exit -> champion), so a
single bar communicates how far a team is likely to go. These finish
probabilities are derived from the cumulative reach-stage probabilities via
`report.outcome_distribution()`; group-position probabilities are available via
`report.group_position_distribution()`. The CLI's `simulate --plot PATH` saves
the outcome-distribution chart directly. `plot_title_race` draws a 95%
Monte Carlo confidence interval around each probability by default
(`show_se=True`; see [Tournament simulation](#tournament-simulation)).

## Methodology

### Elo
Ratings start at 1500 and update after each match (replayed chronologically):

```
expected_home = 1 / (1 + 10^((R_away - R_home - home_adv) / 400))
R_home += K * MOV * (actual - expected_home)
```

A margin-of-victory multiplier (`MOV`) dampens blowouts. The continuous expected
score is split into win/draw/loss probabilities via `draw_base * draw_factor`,
where `draw_factor` peaks at an even matchup and vanishes for lopsided fixtures
(a fixed modeling assumption, not fitted). `draw_base` itself is no longer a
fixed constant: it's calibrated by a method-of-moments fit so the model's
*average* predicted draw probability over the training replay matches the
empirical historical draw rate, clipped to a sane range
(`EloParams.fitted_draw_base_bounds`; fits to ~0.36 on the currently bundled
data, vs. the old fixed 0.27).

### Poisson
Per-team attack and defense strengths are fit by weighted Poisson maximum
likelihood (recency-weighted, with ridge shrinkage toward the mean):

```
log E[goals] = intercept + home_adv * is_home + attack[scorer] - defense[opponent]
```

Goals for the two teams start from independent Poisson variables, then get a
Dixon-Coles correction (Dixon & Coles, 1997): a factor tau, controlled by a
single fitted parameter rho, reweights the four low-scoring cells of the joint
scoreline distribution (0-0, 1-0, 0-1, 1-1) to correct the well-known tendency
of independent Poissons to under/over-estimate draws and low-scoring results.
rho is fit by profile likelihood over the historical low-scoring matches,
bounded (`PoissonParams.rho_bounds`, default ±0.2) to keep every tau factor
well-defined; it fits to rho ≈ -0.06 on the currently bundled data (`wcpredict
train` then check `poisson.rho`, or see notebook 03's printed value). The
correction can be disabled via `PoissonParams.dixon_coles`. This yields a full
scoreline distribution, which drives goal-difference tiebreakers and realistic
knockout results (extra time as a scaled-down independent Poisson -- the
Dixon-Coles correction only models 90-minute dependence -- then a
penalty-shootout lottery weighted slightly toward the stronger side).

### Tournament simulation
Each Monte Carlo run plays all 12 groups (with configurable tiebreakers:
points, goal difference, goals for, head-to-head, then a random drawing of
lots), ranks the third-placed teams to pick the best 8, seeds the 32 qualifiers
into a balanced single-elimination bracket, and plays it out. Host nations
(USA, Mexico, Canada) receive home advantage. Aggregating thousands of runs
yields each team's probability of advancing and reaching each stage.

**Conditioning on results already played.** `TournamentSimulator.run()` (and
`--from-scratch`) is the pre-tournament view: it replays the entire event and
is the right model *before a ball is kicked*. Once the tournament is under way
that view is misleading -- it keeps handing eliminated sides a title chance,
because it never learns they are out. `run_conditioned(state)` fixes this:
`wcpredictor.data.load_tournament_state` reads the vendored 2026 fixtures and
distils how far each team has actually gone plus the current *frontier* round;
the simulator then treats every settled round as fact (probability exactly 0 or
1) and rolls dice only from the frontier onward. Eliminated teams get a 0%
title chance, and the field narrows to the teams still alive. This is the
default for both the CLI `simulate` command and notebook 04. (Only the knockout
phase is conditioned incrementally; a partially played group stage is not
supported, since the 2026 group stage is complete.)

Every reported probability is a Monte Carlo estimate over `n` simulated
tournaments, so it carries sampling noise: the binomial standard error
`se = sqrt(p(1-p)/n)`, available via `report.standard_errors()`. `wcpredict
simulate` prints this as a one-line footer (±1 s.e. on the win probability at
the top of the table), and `viz.plot_title_race` draws it as a 95%
confidence-interval error bar on every team by default.

> Note: the knockout bracket uses a seeding-based pairing (a deterministic,
> balanced approximation) rather than FIFA's exact third-place slotting table,
> which depends on which groups the qualifying third-placed teams come from.

### Evaluation
`wcpredict backtest` runs a walk-forward backtest reporting accuracy, multiclass
log-loss, and the ranked probability score (RPS, the standard metric for ordered
football outcomes; lower is better). `backtest(..., collect_predictions=True)`
additionally records every scored match's predicted probabilities, which
`wcpredictor.evaluation.calibration_table` bins into a reliability-diagram
table (predicted probability vs. empirical frequency, per outcome) --
visualized via `viz.plot_calibration` (see notebook 03).

## Configuration

All hyperparameters live in [`src/wcpredictor/config.py`](src/wcpredictor/config.py)
(Elo K-factor and home advantage, Poisson recency half-life and shrinkage,
number of simulations, tiebreaker order, random seed). Build a custom `Config`
in a notebook to experiment without touching the rest of the code.

A few knobs worth knowing about: `PoissonParams.dixon_coles` toggles the
low-score correlation correction on/off, with `rho_bounds` as its fitting
range; `EloParams.fitted_draw_base_bounds` clips the fitted draw base (see
[Elo](#elo)). `PoissonParams.half_life_days` defaults to ~10 years (3650
days) rather than the ~5 years you might expect: with real finals-only data
(one tournament every 4 years), a 5-year half-life leaves too few effective
matches to fit ~170 Poisson parameters, so 10 years keeps the last 2-3 World
Cups dominant while older tournaments still contribute.

Saved model artifacts (`data/processed/*.csv`) now embed the fitted
Dixon-Coles rho and Elo draw base alongside the ratings/strengths tables;
artifacts saved by an older version of the code (without those fields) still
load fine (backward compatible).

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

> **Careful with `--refresh` mid-tournament:** while the 2026 finals are in
> progress, re-downloading the source JSON pulls in whatever knockout results
> have been played *since* the vendored snapshot -- including results the
> model would otherwise be "predicting". That breaks reproducibility of any
> predictions you've already published. Stick to the default (no `--refresh`,
> uses the vendored snapshot) unless you specifically want the latest results.

A live football API could be wired in behind `data/loader.py` by writing a
fetcher that produces the same `matches.csv` schema.

## Development

```bash
pytest          # run the test suite
```

## License

MIT
