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
cover every World Cup finals from 1930 through the **completed** 2026 tournament
(the full group stage and every knockout round, up to and including the final
Spain won), and the bundled group stage is the **actual 2026 draw**.

> **Coverage note:** the source only includes World Cup *finals* (no qualifiers
> or friendlies). Four teams make their World Cup debut in 2026 -- Cape Verde,
> Curaçao, Jordan and Uzbekistan -- so they have no pre-2026 finals history and
> their ratings rest entirely on their 2026 games. A team only falls back to the
> models' *default* ratings when it has no matches at all; now that the full
> 2026 tournament has been played, every qualifier has a record, so none
> currently do. `notebooks/01_data_exploration.ipynb` recomputes
> that no-history list live against whatever data is bundled (it is empty as of
> the currently vendored data). To fold in more matches, see
> [Using your own data](#using-your-own-data).

## Project layout

```
world-cup-predictor/
├── data/
│   ├── raw/                  bundled CSVs (matches, teams, groups)
│   ├── source/               vendored openfootball worldcup.json (per year)
│   └── processed/            saved model artifacts (generated)
├── notebooks/                analysis notebooks (01-09 + 9x appendices)
├── scripts/
│   ├── fetch_worldcup_data.py build raw CSVs from openfootball data
│   ├── generate_seed_data.py  regenerate synthetic CSVs (offline fallback)
│   └── build_notebooks.py     regenerate the notebooks
├── src/wcpredictor/
│   ├── config.py             paths + model hyperparameters
│   ├── data/                 loader.py, preprocess.py, tournament_state.py
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
the vendored 2026 fixtures): every settled round is locked in and only what is
still undecided is rolled forward, so eliminated teams correctly show a 0%
title chance. The vendored data is now the **completed** tournament, so there is
no bracket left to roll forward -- `simulate` rewinds to the final and projects
that last match (`wcpredict simulate --top 2`):

```
The vendored 2026 tournament is complete; projecting the final as the live frontier.
Conditioning on results through the final: 2 teams still alive. Simulating 10,000 times from the current bracket...

  #  Team             Grp     Win   Final    Semi   Last8     Adv
  1  Spain              H   60.4%  100.0%  100.0%  100.0%  100.0%
  2  Argentina          J   39.6%  100.0%  100.0%  100.0%  100.0%
```

The model made Spain a clear favourite for the final, and Spain duly won it 1-0
after extra time. Pass `--from-scratch` for the pre-tournament projection that
replays the whole event (all 48 teams contend, ignoring who was later knocked
out).

## Notebooks

```bash
jupyter lab notebooks/
```

- `01_data_exploration.ipynb` - inspect the seed data and outcome distributions.
- `02_elo_ratings.ipynb` - build and visualize Elo ratings.
- `03_poisson_model.ipynb` - attack/defense strengths, scoreline heatmaps, backtest.

### Per-phase prediction notebooks (04-09)

Notebooks `04`-`09` are a series, one per tournament phase, that answer "what
did we predict, and how right were we?" at each stage:

- `04_group_stage.ipynb` - **pre-tournament**: per-fixture win/draw/loss odds
  for every group game and each team's advancement probability.
- `05_round_of_32.ipynb` - the 16 R32 ties and everything downstream.
- `06_round_of_16.ipynb` - the 8 R16 ties and downstream.
- `07_quarterfinals.ipynb` - the 4 QF ties and downstream.
- `08_semifinals.ipynb` - the 2 semifinal ties (both won by the underdog).
- `09_final.ipynb` - the final itself, then the champion crowned: the settled
  scoreline, the third-place match, and the title odds the model gave the
  eventual winner going in.

Two ideas run through every one:

1. **Point-in-time training (no look-ahead).** Each notebook trains on *only*
   the matches played strictly before its round began (via
   `wcpredictor.data.phase_start_dates`), so a snapshot never "knows" a result
   it is about to predict. The training-set sizes step up round by round
   (964 → 1036 → 1052 → 1060 → 1064 → 1067 matches).
2. **Conditioning + scoring.** Each round is projected with
   `load_tournament_state(config, as_of_stage=...)`, which *rewinds* the
   fixtures to the start of that round, followed by `sim.run_conditioned`. For
   a knockout tie, a team's `p_<next stage>` **is** its probability of winning
   that tie (the two sides sum to 100%). Rounds that have since been played are
   graded against reality with `wcpredictor.data.actual_knockout_ties`
   (accuracy / Brier / log-loss); the still-to-come round shows live
   predictions only.

The series was built to extend a round at a time as the tournament progressed,
and `09_final.ipynb` now closes it out on the same recipe (`notebook_final` in
`scripts/build_notebooks.py` reuses the shared knockout-phase builder with the
final as the frontier, then appends the champion crowning). With the final
played, the whole 2026 event is graded end to end.

### Reference analyses (9x)

Notebooks numbered `90+` are appendices, deliberately out of the phase
series' numbering:

- `90_knockout_scores.ipynb` - every quarterfinal-onward match of the last
  five completed World Cups (2006-2022) plus the already-played 2026 rounds:
  the distribution of full-time scores and of scores after extra time, how
  often 90 minutes fails to settle a tie, and an empirical check of the
  simulator's extra-time scaling assumption (`EXTRA_TIME_FRACTION`). Unplayed
  2026 fixtures are skipped, so the sample grows as results are refreshed.

Rebuild them all (they are generated artifacts -- edit
`scripts/build_notebooks.py`, not the `.ipynb` files) with `python
scripts/build_notebooks.py`.

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
well-defined; it fits to rho ≈ -0.09 on the currently bundled data (`wcpredict
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
default for the CLI `simulate` command and the knockout phase notebooks
(05-09); the group-stage notebook (04) instead uses the from-scratch `run()`
as its pre-tournament view. (Only the knockout
phase is conditioned incrementally; a partially played group stage is not
supported, since the 2026 group stage is complete.)

Every reported probability is a Monte Carlo estimate over `n` simulated
tournaments, so it carries sampling noise: the binomial standard error
`se = sqrt(p(1-p)/n)`, available via `report.standard_errors()`. `wcpredict
simulate` prints this as a one-line footer (±1 s.e. on the win probability at
the top of the table), and `viz.plot_title_race` draws it as a 95%
confidence-interval error bar on every team by default.

> Note: this only applies to the from-scratch projection, whose knockout
> bracket uses a seeding-based pairing (a deterministic, balanced approximation)
> rather than FIFA's exact third-place slotting table, which depends on which
> groups the qualifying third-placed teams come from. The conditioned default
> sidesteps the approximation once the knockout is under way -- it plays the
> *actual* bracket read from the fixtures.

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

> **Note on `--refresh`:** the 2026 tournament is complete and the vendored
> snapshot already holds every result, so `--refresh` is effectively a no-op
> for it now. It mattered *during* the tournament: re-downloading then pulled
> in knockout results played *since* the snapshot -- including ones a phase
> notebook is meant to be "predicting" -- which breaks reproducibility of any
> predictions you'd already published. Stick to the default (no `--refresh`,
> uses the vendored snapshot) unless you deliberately want to re-pull.

A live football API could be wired in behind `data/loader.py` by writing a
fetcher that produces the same `matches.csv` schema.

## Adding more models

The Towards Data Science article in [References](#references) builds 11
predictors for this same tournament -- ratings (Elo, Colley, PageRank), goal
models (Poisson GLM, negative binomial), outcome classifiers (logistic
regression, k-NN, random forest, XGBoost, a small neural net) and a betting-
market benchmark -- and finds they crown four different champions. This
project already covers its Elo and Poisson families; the rest slot into the
existing architecture at one of two levels:

1. **Outcome models** (Colley, PageRank, the ML classifiers): subclass
   `MatchModel` in `src/wcpredictor/models/base.py` -- implement `fit()` over
   the `build_training_matches` frame and `predict_match()` returning a
   `MatchPrediction` (win/draw/loss probabilities). That alone plugs the model
   into `wcpredict match`, the walk-forward `backtest`, calibration tables and
   `viz.plot_model_comparison`, so it can be scored head-to-head against Elo
   and Poisson on the bundled data. The classifiers need features; the
   article's three (strength gap, combined strength, knockout flag) can all be
   derived from an Elo fit plus the match metadata.
2. **Goal models** (e.g. the negative binomial, which relaxes Poisson's
   mean = variance assumption): additionally expose `expected_goals()` and
   `sample_score()` -- the two methods `simulate_match` actually calls. Any
   model with those can be handed to `TournamentSimulator` in place of
   `PoissonModel` and drive the full Monte Carlo (group tiebreakers, extra
   time, penalties), the conditioned per-phase notebooks included.

Outcome-only models can't drive the simulator (it needs scorelines for goal
difference and extra time), so classifiers stop at level 1 unless paired with
a goal model. Two cheap, high-value additions from the article's conclusions:
a simple **ensemble** (a `MatchModel` that averages the probabilities of
several fitted members -- "a simple ensemble usually beats most of its
members") and a **betting-market benchmark** (a pseudo-model that reads
de-vigged odds from a CSV) as the yardstick every model must beat in
`backtest`. Note the small-data caveat: on finals-only history (~1k matches)
the article found simpler models beat the flexible ones (XGBoost, neural
nets), which overfit -- if you go down the ML route, first broaden the
training set (see [Using your own data](#using-your-own-data)).

## Development

```bash
pytest          # run the test suite
```

## References

- [I Built 11 Models to Predict the 2026 World Cup — They Crown Four Different
  Champions](https://towardsdatascience.com/i-built-11-models-to-predict-the-2026-world-cup-they-crown-four-different-champions/)
  (Towards Data Science) -- a survey of approaches to the same prediction
  problem, including Elo- and Poisson-based models like the ones used here;
  see [Adding more models](#adding-more-models) for how the other nine fit
  into this codebase.
- Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market* -- the low-scoring correction
  applied to the Poisson model (see [Methodology](#methodology)).
- [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json)
  -- the bundled historical match data (CC0).

## License

MIT
