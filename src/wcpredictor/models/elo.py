"""Elo rating model for international football.

Classic Elo adapted with a home-advantage term and an optional
margin-of-victory multiplier. Ratings are learned by replaying matches in
chronological order. The continuous Elo "expected score" is converted into
win/draw/loss probabilities via a simple, interpretable draw model whose
scale (``fitted_draw_base``) is calibrated from the same replay, by a method-
of-moments fit against the empirical draw rate (see ``fit``).
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from wcpredictor.config import Config, default_config
from wcpredictor.models.base import MatchModel, MatchPrediction


class EloModel(MatchModel):
    """Elo ratings with home advantage and W/D/L probability output."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config()
        self.params = self.config.elo
        self.ratings: Optional[Dict[str, float]] = None
        # Data-driven draw base fitted in ``fit``; falls back to the prior in
        # ``params.draw_base`` while None (e.g. before fitting).
        self.fitted_draw_base: Optional[float] = None
        # Teams already warned about being unseen (warn once per instance).
        self._warned_unseen: Set[str] = set()
        # Per-match rating trajectory, only populated by fit(track_history=True).
        self.history: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ fit
    def fit(
        self, training_matches: pd.DataFrame, track_history: bool = False
    ) -> "EloModel":
        """Replay matches chronologically, updating ratings in place.

        Alongside the rating walk-forward, calibrates ``fitted_draw_base`` by
        method of moments using PRE-match ratings (true walk-forward, no
        lookahead): ``predict_match`` turns the Elo expected score into a
        draw probability via ``draw_base * draw_factor`` where
        ``draw_factor = 1 - 2*|exp_home - 0.5|`` peaks at 1 for an even match
        and vanishes for a near-certain result. Choosing
        ``draw_base = n_draws / sum(draw_factor)`` makes the *mean* predicted
        draw probability over these matches equal the empirical draw rate
        exactly (the moment condition); the triangular shape of
        ``draw_factor`` itself stays an assumption, not something fit here. A
        logistic draw curve (fitting both scale and shape) was considered and
        rejected: with only 984 finals matches and a modest draw count, there
        isn't enough signal to fit the curve's shape reliably, only its mean.

        Set ``track_history=True`` to additionally record every team's rating
        after each match it plays into ``self.history`` (``date``, ``team``,
        ``rating`` rows), e.g. for plotting rating trajectories over time.
        Off by default: it doubles the rows retained for a full replay, and
        callers that don't need it (simulation, backtesting) shouldn't pay
        for it. Purely additive bookkeeping -- it does not affect the fitted
        ratings or ``fitted_draw_base``.
        """
        params = self.params
        ratings: Dict[str, float] = {}
        draw_factor_sum = 0.0
        n_draws = 0
        history_rows: List[Tuple] = []

        df = training_matches.sort_values("date")
        for row in df.itertuples(index=False):
            home, away = row.home_team, row.away_team
            ratings.setdefault(home, params.initial_rating)
            ratings.setdefault(away, params.initial_rating)

            neutral = bool(getattr(row, "neutral", False))
            exp_home = self._expected_score(
                ratings[home], ratings[away], neutral
            )

            # Draw-base calibration, using the PRE-match rating's expected
            # score computed just above (ratings are updated below).
            draw_factor_sum += 1.0 - 2.0 * abs(exp_home - 0.5)
            if row.home_score == row.away_score:
                n_draws += 1

            if row.home_score > row.away_score:
                score_home = 1.0
            elif row.home_score < row.away_score:
                score_home = 0.0
            else:
                score_home = 0.5

            mov = self._mov_multiplier(
                row.home_score - row.away_score,
                ratings[home] - ratings[away],
            )
            delta = params.k_factor * mov * (score_home - exp_home)
            ratings[home] += delta
            ratings[away] -= delta

            if track_history:
                history_rows.append((row.date, home, ratings[home]))
                history_rows.append((row.date, away, ratings[away]))

        self.ratings = ratings
        self.history = (
            pd.DataFrame(history_rows, columns=["date", "team", "rating"])
            if track_history
            else None
        )
        if draw_factor_sum > 0:
            self.fitted_draw_base = float(
                np.clip(
                    n_draws / draw_factor_sum, *params.fitted_draw_base_bounds
                )
            )
        return self

    # -------------------------------------------------------------- predict
    def _expected_score(
        self, rating_home: float, rating_away: float, neutral: bool
    ) -> float:
        adj_home = rating_home + (0.0 if neutral else self.params.home_advantage)
        return 1.0 / (1.0 + 10 ** ((rating_away - adj_home) / self.params.scale))

    def _mov_multiplier(self, goal_diff: int, rating_diff: float) -> float:
        """Margin-of-victory multiplier (World-Football-Elo style).

        ``goal_diff`` and ``rating_diff`` are both home-minus-away. The
        dampening term uses the *winner's* rating advantage (winner minus
        loser), so it is asymmetric by design: a heavy favourite winning big
        is dampened (its denominator grows, the win counts for less) while an
        underdog winning big is amplified (its denominator shrinks, the upset
        counts for more). Using ``abs(rating_diff)`` would collapse both cases
        to the same multiplier, contradicting that intent -- so the sign is
        taken from ``goal_diff`` (who actually won) here.
        """
        if self.params.mov_factor == 0:
            return 1.0
        margin = abs(goal_diff)
        if margin <= 1:
            base = 1.0
        else:
            damp = self.params.mov_dampening
            # Winner-minus-loser rating gap: positive when the favourite won,
            # negative on an upset.
            winner_adv = rating_diff if goal_diff > 0 else -rating_diff
            # Floor the denominator so an extreme upset can't drive it to zero
            # or flip its sign (never reached by real football rating gaps).
            denom = max(
                self.params.mov_rating_slope * winner_adv + damp, 1e-6
            )
            base = math.log(margin + 1.0) * (damp / denom)
        return 1.0 + self.params.mov_factor * (base - 1.0)

    def rating(self, team: str) -> float:
        """Current rating for a team (initial rating if unseen)."""
        self._check_fitted("ratings")
        if team not in self.ratings:
            if team not in self._warned_unseen:
                warnings.warn(
                    f"Team '{team}' not seen in training data; using initial "
                    f"Elo rating {self.params.initial_rating:.0f}.",
                    UserWarning,
                )
                self._warned_unseen.add(team)
            return self.params.initial_rating
        return self.ratings[team]

    def predict_match(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> MatchPrediction:
        self._check_fitted("ratings")
        exp_home = self._expected_score(
            self.rating(home_team), self.rating(away_team), neutral
        )
        # Draw model: draw probability peaks at an even matchup and vanishes as
        # the expected score approaches a certain win/loss.
        draw_factor = 1.0 - 2.0 * abs(exp_home - 0.5)
        draw_base = (
            self.fitted_draw_base
            if getattr(self, "fitted_draw_base", None) is not None
            else self.params.draw_base
        )
        p_draw = draw_base * draw_factor
        p_home = exp_home - 0.5 * p_draw
        p_away = (1.0 - exp_home) - 0.5 * p_draw
        # Guard against tiny floating point negatives.
        p_home = max(p_home, 0.0)
        p_away = max(p_away, 0.0)
        total = p_home + p_draw + p_away
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home_win=p_home / total,
            p_draw=p_draw / total,
            p_away_win=p_away / total,
        )

    # --------------------------------------------------------------- export
    def ratings_table(self, teams: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return ratings as a sorted DataFrame, optionally with metadata."""
        self._check_fitted("ratings")
        table = pd.DataFrame(
            {"team": list(self.ratings.keys()),
             "rating": list(self.ratings.values())}
        )
        if teams is not None:
            table = table.merge(teams, on="team", how="left")
        return table.sort_values("rating", ascending=False).reset_index(drop=True)

    def save(self, path=None, teams: Optional[pd.DataFrame] = None) -> None:
        """Persist ratings to CSV, prepending a ``__draw_base__`` meta row."""
        path = path or self.config.elo_ratings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self.ratings_table(teams)
        # NOTE: must be an explicit `is not None` check, not `or` -- `or`
        # would treat a fitted draw base of exactly 0.0 as falsy and silently
        # fall back to the prior. (fitted_draw_base_bounds makes 0.0
        # unreachable in practice, but the check should be correct anyway.)
        draw_base = (
            self.fitted_draw_base
            if getattr(self, "fitted_draw_base", None) is not None
            else self.params.draw_base
        )
        meta = pd.DataFrame({"team": ["__draw_base__"], "rating": [draw_base]})
        pd.concat([meta, table], ignore_index=True).to_csv(path, index=False)

    @classmethod
    def load(cls, path=None, config: Optional[Config] = None) -> "EloModel":
        """Load ratings from a CSV produced by :meth:`save`.

        Tolerates legacy CSVs that predate the ``__``-prefixed meta rows.
        """
        config = config or default_config()
        path = path or config.elo_ratings_path
        df = pd.read_csv(path)
        model = cls(config)
        meta_mask = df["team"].astype(str).str.startswith("__")
        meta = df[meta_mask].set_index("team")
        if "__draw_base__" in meta.index:
            model.fitted_draw_base = float(meta.loc["__draw_base__", "rating"])
        ratings_df = df[~meta_mask]
        model.ratings = dict(zip(ratings_df["team"], ratings_df["rating"]))
        return model
