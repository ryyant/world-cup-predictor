"""Simulate a single match, including knockout extra time and penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Fraction of a 90-minute match represented by 30 minutes of extra time.
EXTRA_TIME_FRACTION = 30.0 / 90.0


@dataclass
class MatchResult:
    """Outcome of a simulated match."""

    home_team: str
    away_team: str
    home_score: int
    away_score: int
    winner: Optional[str] = None  # None means a draw (group stage only)
    extra_time: bool = False
    penalties: bool = False

    @property
    def is_draw(self) -> bool:
        return self.winner is None


def simulate_match(
    model,
    home_team: str,
    away_team: str,
    rng: np.random.Generator,
    neutral: bool = True,
    knockout: bool = False,
    penalty_edge_weight: float = 0.5,
) -> MatchResult:
    """Simulate one match.

    ``model`` must expose ``sample_score(home, away, rng, neutral)`` and
    ``expected_goals(home, away, neutral)`` (the :class:`PoissonModel` API).

    In the group stage (``knockout=False``) a draw is a valid result. In a
    knockout (``knockout=True``) a tie is broken with simulated extra time and,
    if still level, a penalty shootout weighted slightly toward the stronger
    side.
    """
    home_score, away_score = model.sample_score(
        home_team, away_team, rng, neutral=neutral
    )

    if not knockout or home_score != away_score:
        winner = None
        if home_score > away_score:
            winner = home_team
        elif away_score > home_score:
            winner = away_team
        return MatchResult(
            home_team, away_team, home_score, away_score, winner
        )

    # Knockout tie -> extra time (scaled-down Poisson on the same strengths).
    # Unlike sample_score's 90-minute draw, this stays independent Poisson:
    # a Dixon-Coles model only captures 90-minute low-score dependence, not
    # extra time's much smaller sample of goals.
    lam_home, lam_away = model.expected_goals(home_team, away_team, neutral)
    et_home = int(rng.poisson(lam_home * EXTRA_TIME_FRACTION))
    et_away = int(rng.poisson(lam_away * EXTRA_TIME_FRACTION))
    home_score += et_home
    away_score += et_away

    if home_score != away_score:
        winner = home_team if home_score > away_score else away_team
        return MatchResult(
            home_team, away_team, home_score, away_score, winner,
            extra_time=True,
        )

    # Penalty shootout: probability weighted by relative attacking strength,
    # pulled toward 50/50 to reflect the lottery nature of shootouts.
    edge = lam_home / (lam_home + lam_away) if (lam_home + lam_away) > 0 else 0.5
    p_home_pen = 0.5 + penalty_edge_weight * (edge - 0.5)  # scale the edge
    winner = home_team if rng.random() < p_home_pen else away_team
    return MatchResult(
        home_team, away_team, home_score, away_score, winner,
        extra_time=True, penalties=True,
    )
