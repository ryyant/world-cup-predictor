"""Poisson goals model.

Estimates per-team attack and defense strengths via weighted Poisson maximum
likelihood:

    log E[goals] = intercept + home_advantage * is_home
                   + attack[scoring_team] - defense[conceding_team]

Goals for the two teams are then treated as independent Poisson variables,
yielding a full scoreline distribution. This is what powers goal-difference
tiebreakers in the group stage and realistic knockout results (incl. the
possibility of extra time / penalties handled in the simulation layer).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson as poisson_dist

from wcpredictor.config import Config, default_config
from wcpredictor.models.base import MatchModel, MatchPrediction


class PoissonModel(MatchModel):
    """Weighted Poisson attack/defense model with scoreline sampling."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config()
        self.params = self.config.poisson
        self.attack: Optional[Dict[str, float]] = None
        self.defense: Optional[Dict[str, float]] = None
        self.intercept: float = 0.0
        self.home_advantage: float = 0.0
        self._teams: Optional[list] = None

    # ------------------------------------------------------------------ fit
    def fit(self, training_matches: pd.DataFrame) -> "PoissonModel":
        df = training_matches
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = df["home_team"].map(index).to_numpy()
        ai = df["away_team"].map(index).to_numpy()
        hg = df["home_score"].to_numpy(dtype=float)
        ag = df["away_score"].to_numpy(dtype=float)
        not_neutral = (~df["neutral"].to_numpy(dtype=bool)).astype(float)
        if "recency_weight" in df.columns:
            w = df["recency_weight"].to_numpy(dtype=float)
        else:
            w = np.ones(len(df))

        shrink = self.params.shrinkage

        # Parameter layout: [attack(n), defense(n), intercept, home_adv].
        def unpack(x):
            return x[:n], x[n:2 * n], x[2 * n], x[2 * n + 1]

        def neg_ll_and_grad(x):
            attack, defense, mu, ha = unpack(x)
            # Expected goals for each observation.
            log_lam_h = mu + ha * not_neutral + attack[hi] - defense[ai]
            log_lam_a = mu + attack[ai] - defense[hi]
            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)

            ll = np.sum(w * (hg * log_lam_h - lam_h)) + np.sum(
                w * (ag * log_lam_a - lam_a)
            )
            penalty = shrink * (np.sum(attack ** 2) + np.sum(defense ** 2))
            obj = -(ll - penalty)

            # Gradients (of +LL); residuals weighted.
            rh = w * (hg - lam_h)
            ra = w * (ag - lam_a)

            g_attack = np.zeros(n)
            g_defense = np.zeros(n)
            np.add.at(g_attack, hi, rh)
            np.add.at(g_attack, ai, ra)
            np.add.at(g_defense, ai, -rh)
            np.add.at(g_defense, hi, -ra)

            g_mu = np.sum(rh) + np.sum(ra)
            g_ha = np.sum(rh * not_neutral)

            # Convert to gradient of objective (-LL + penalty).
            g_attack = -g_attack + 2 * shrink * attack
            g_defense = -g_defense + 2 * shrink * defense
            grad = np.concatenate(
                [g_attack, g_defense, [-g_mu], [-g_ha]]
            )
            return obj, grad

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = np.log(max(self.params.base_goals, 1e-3))
        x0[2 * n + 1] = np.log(self.params.home_factor)

        result = optimize.minimize(
            neg_ll_and_grad, x0, jac=True, method="L-BFGS-B",
            options={"maxiter": 500},
        )
        attack, defense, mu, ha = unpack(result.x)

        # Center strengths for identifiability/interpretability; fold the means
        # back into the intercept so predictions are unchanged.
        a_mean = attack.mean()
        d_mean = defense.mean()
        attack = attack - a_mean
        defense = defense - d_mean
        mu = mu + a_mean - d_mean

        self.attack = {t: float(attack[i]) for t, i in index.items()}
        self.defense = {t: float(defense[i]) for t, i in index.items()}
        self.intercept = float(mu)
        self.home_advantage = float(ha)
        self._teams = teams
        return self

    # -------------------------------------------------------------- predict
    def expected_goals(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> Tuple[float, float]:
        """Return expected goals for (home, away)."""
        self._check_fitted("attack")
        ha = 0.0 if neutral else self.home_advantage
        a_home = self.attack.get(home_team, 0.0)
        a_away = self.attack.get(away_team, 0.0)
        d_home = self.defense.get(home_team, 0.0)
        d_away = self.defense.get(away_team, 0.0)
        lam_home = np.exp(self.intercept + ha + a_home - d_away)
        lam_away = np.exp(self.intercept + a_away - d_home)
        return float(lam_home), float(lam_away)

    def scoreline_matrix(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> np.ndarray:
        """Joint probability matrix P[i, j] of home i goals, away j goals."""
        lam_home, lam_away = self.expected_goals(home_team, away_team, neutral)
        max_g = self.params.max_goals
        goals = np.arange(max_g + 1)
        p_home = poisson_dist.pmf(goals, lam_home)
        p_away = poisson_dist.pmf(goals, lam_away)
        matrix = np.outer(p_home, p_away)
        total = matrix.sum()
        if total > 0:
            matrix = matrix / total  # renormalize truncated tail
        return matrix

    def predict_match(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> MatchPrediction:
        matrix = self.scoreline_matrix(home_team, away_team, neutral)
        p_home = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())
        total = p_home + p_draw + p_away
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home_win=p_home / total,
            p_draw=p_draw / total,
            p_away_win=p_away / total,
        )

    def most_likely_score(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> Tuple[int, int]:
        matrix = self.scoreline_matrix(home_team, away_team, neutral)
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        return int(i), int(j)

    def sample_score(
        self,
        home_team: str,
        away_team: str,
        rng: np.random.Generator,
        neutral: bool = False,
    ) -> Tuple[int, int]:
        """Sample a single scoreline using independent Poisson draws."""
        lam_home, lam_away = self.expected_goals(home_team, away_team, neutral)
        return int(rng.poisson(lam_home)), int(rng.poisson(lam_away))

    # --------------------------------------------------------------- export
    def strengths_table(self, teams: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self._check_fitted("attack")
        table = pd.DataFrame(
            {
                "team": list(self.attack.keys()),
                "attack": list(self.attack.values()),
                "defense": [self.defense[t] for t in self.attack],
            }
        )
        table["net_strength"] = table["attack"] + table["defense"]
        if teams is not None:
            table = table.merge(teams, on="team", how="left")
        return table.sort_values("net_strength", ascending=False).reset_index(
            drop=True
        )

    def save(self, path=None, teams: Optional[pd.DataFrame] = None) -> None:
        path = path or self.config.poisson_strengths_path
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self.strengths_table(teams)
        # Stash global params as attributes via extra columns on first row group.
        table.attrs["intercept"] = self.intercept
        table.attrs["home_advantage"] = self.home_advantage
        meta = pd.DataFrame(
            {
                "team": ["__intercept__", "__home_advantage__"],
                "attack": [self.intercept, self.home_advantage],
                "defense": [0.0, 0.0],
                "net_strength": [0.0, 0.0],
            }
        )
        pd.concat([meta, table], ignore_index=True).to_csv(path, index=False)

    @classmethod
    def load(cls, path=None, config: Optional[Config] = None) -> "PoissonModel":
        config = config or default_config()
        path = path or config.poisson_strengths_path
        df = pd.read_csv(path)
        model = cls(config)
        meta = df.set_index("team")
        model.intercept = float(meta.loc["__intercept__", "attack"])
        model.home_advantage = float(meta.loc["__home_advantage__", "attack"])
        teams_df = df[~df["team"].str.startswith("__")]
        model.attack = dict(zip(teams_df["team"], teams_df["attack"]))
        model.defense = dict(zip(teams_df["team"], teams_df["defense"]))
        model._teams = list(teams_df["team"])
        return model
