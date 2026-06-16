"""Elo rating model for international football.

Classic Elo adapted with a home-advantage term and an optional
margin-of-victory multiplier. Ratings are learned by replaying matches in
chronological order. The continuous Elo "expected score" is converted into
win/draw/loss probabilities via a simple, interpretable draw model.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import pandas as pd

from wcpredictor.config import Config, default_config
from wcpredictor.models.base import MatchModel, MatchPrediction


class EloModel(MatchModel):
    """Elo ratings with home advantage and W/D/L probability output."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config()
        self.params = self.config.elo
        self.ratings: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------ fit
    def fit(self, training_matches: pd.DataFrame) -> "EloModel":
        """Replay matches chronologically, updating ratings in place."""
        params = self.params
        ratings: Dict[str, float] = {}

        df = training_matches.sort_values("date")
        for row in df.itertuples(index=False):
            home, away = row.home_team, row.away_team
            ratings.setdefault(home, params.initial_rating)
            ratings.setdefault(away, params.initial_rating)

            neutral = bool(getattr(row, "neutral", False))
            exp_home = self._expected_score(
                ratings[home], ratings[away], neutral
            )

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

        self.ratings = ratings
        return self

    # -------------------------------------------------------------- predict
    def _expected_score(
        self, rating_home: float, rating_away: float, neutral: bool
    ) -> float:
        adj_home = rating_home + (0.0 if neutral else self.params.home_advantage)
        return 1.0 / (1.0 + 10 ** ((rating_away - adj_home) / self.params.scale))

    def _mov_multiplier(self, goal_diff: int, rating_diff: float) -> float:
        if self.params.mov_factor == 0:
            return 1.0
        margin = abs(goal_diff)
        if margin <= 1:
            base = 1.0
        else:
            # FIFA-style dampening so blowouts count less, and so a favourite
            # winning big does not inflate as much as an underdog doing so.
            base = math.log(margin + 1.0) * (
                2.2 / (0.001 * abs(rating_diff) + 2.2)
            )
        return 1.0 + self.params.mov_factor * (base - 1.0)

    def rating(self, team: str) -> float:
        """Current rating for a team (initial rating if unseen)."""
        self._check_fitted("ratings")
        return self.ratings.get(team, self.params.initial_rating)

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
        p_draw = self.params.draw_base * draw_factor
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
        """Persist ratings to CSV."""
        path = path or self.config.elo_ratings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ratings_table(teams).to_csv(path, index=False)

    @classmethod
    def load(cls, path=None, config: Optional[Config] = None) -> "EloModel":
        """Load ratings from a CSV produced by :meth:`save`."""
        config = config or default_config()
        path = path or config.elo_ratings_path
        df = pd.read_csv(path)
        model = cls(config)
        model.ratings = dict(zip(df["team"], df["rating"]))
        return model
