"""Shared interface for match-prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchPrediction:
    """Win/draw/loss probabilities for a single fixture (home perspective)."""

    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float

    def __post_init__(self) -> None:
        total = self.p_home_win + self.p_draw + self.p_away_win
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"Probabilities must sum to 1, got {total:.4f} for "
                f"{self.home_team} vs {self.away_team}"
            )

    @property
    def most_likely(self) -> str:
        """Return 'H', 'D', or 'A' for the highest-probability outcome."""
        options = {"H": self.p_home_win, "D": self.p_draw, "A": self.p_away_win}
        return max(options, key=options.get)


class MatchModel(ABC):
    """Abstract base class for models that predict a single match outcome."""

    @abstractmethod
    def fit(self, training_matches) -> "MatchModel":
        """Estimate model parameters from a training-match DataFrame."""

    @abstractmethod
    def predict_match(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> MatchPrediction:
        """Return win/draw/loss probabilities for a fixture."""

    def _check_fitted(self, attr: str) -> None:
        if getattr(self, attr, None) is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call .fit() first."
            )
