"""Poisson goals model.

Estimates per-team attack and defense strengths via weighted Poisson maximum
likelihood:

    log E[goals] = intercept + home_advantage * is_home
                   + attack[scoring_team] - defense[conceding_team]

Goals for the two teams start from independent Poisson variables, then get a
Dixon-Coles correction (Dixon & Coles, 1997) applied to the four low-score
cells (0-0, 0-1, 1-0, 1-1): real football has more 0-0/1-1 draws and fewer
1-0/0-1 results than independence predicts, because defensive, cautious
matches suppress both teams' goals together. The correction is a single
fitted parameter rho (see ``fit``) and can be disabled via
``PoissonParams.dixon_coles``. This yields a full scoreline distribution,
which is what powers goal-difference tiebreakers in the group stage and
realistic knockout results (incl. the possibility of extra time / penalties
handled in the simulation layer).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Set, Tuple

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
        # Dixon-Coles low-score correlation parameter (see module docstring
        # and ``fit``); 0.0 means "no correction" (plain independent Poisson).
        self.rho: float = 0.0
        # Teams already warned about being unseen (warn once per instance):
        # expected_goals is called ~10^6 times per simulation run.
        self._warned_unseen: Set[str] = set()
        # Cache of (scoreline matrix, flattened cumsum) keyed by
        # (home_team, away_team, neutral), used by sample_score for fast
        # inverse-CDF draws. Bounded: at most (teams x teams x 2) entries.
        self._matrix_cache: Dict[
            Tuple[str, str, bool], Tuple[np.ndarray, np.ndarray]
        ] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, training_matches: pd.DataFrame) -> "PoissonModel":
        """Estimate attack/defense strengths, home advantage and rho.

        Maximises the (optionally recency-weighted) Poisson log-likelihood
        with a ridge penalty on the strengths via L-BFGS-B, centers the fitted
        strengths for identifiability, then fits the Dixon-Coles ``rho`` by
        profile likelihood (see :meth:`_fit_rho`). Uses a ``recency_weight``
        column if present, else weights every match equally. Returns ``self``.
        """
        # Strengths (and therefore every cached scoreline matrix) are about
        # to change; drop anything sample_score cached from a previous fit.
        self._matrix_cache.clear()

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
        if not result.success:
            warnings.warn(
                f"Poisson MLE did not fully converge: {result.message}. "
                "Strengths may be unreliable; consider more data or a longer "
                "optimizer run.",
                RuntimeWarning,
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

        # Fit the Dixon-Coles rho by profile likelihood: attack/defense/
        # intercept/home_advantage are held fixed at their MLE above, and rho
        # is chosen to best explain the observed low-score results.
        log_lam_h = mu + ha * not_neutral + attack[hi] - defense[ai]
        log_lam_a = mu + attack[ai] - defense[hi]
        self.rho = self._fit_rho(hg, ag, np.exp(log_lam_h), np.exp(log_lam_a), w)
        return self

    def _fit_rho(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        lam_home: np.ndarray,
        lam_away: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Fit the Dixon-Coles rho by profile likelihood on low-score matches.

        Only matches with both scores <= 1 carry information about rho (tau
        is 1, a no-op, everywhere else), so the objective and its bounds are
        built entirely from those. The search interval is ``rho_bounds``
        intersected with the region where every observed tau factor stays
        strictly positive, shrunk slightly inward so log(tau) never touches
        zero (i.e. -inf) at the boundary.
        """
        if not self.params.dixon_coles:
            return 0.0

        mask00 = (home_goals == 0) & (away_goals == 0)
        mask01 = (home_goals == 0) & (away_goals == 1)
        mask10 = (home_goals == 1) & (away_goals == 0)
        mask11 = (home_goals == 1) & (away_goals == 1)
        if not (mask00.any() or mask01.any() or mask10.any() or mask11.any()):
            return 0.0

        lam00, mu00, w00 = lam_home[mask00], lam_away[mask00], weights[mask00]
        lam01, w01 = lam_home[mask01], weights[mask01]
        mu10, w10 = lam_away[mask10], weights[mask10]
        w11 = weights[mask11]

        lower_candidates = [self.params.rho_bounds[0]]
        upper_candidates = [self.params.rho_bounds[1]]
        if mask00.any():
            # tau(0,0) = 1 - lam*mu*rho > 0  =>  rho < 1 / (lam*mu).
            upper_candidates.append(float(np.min(1.0 / (lam00 * mu00))))
        if mask01.any():
            # tau(0,1) = 1 + lam*rho > 0  =>  rho > -1 / lam.
            lower_candidates.append(float(np.max(-1.0 / lam01)))
        if mask10.any():
            # tau(1,0) = 1 + mu*rho > 0  =>  rho > -1 / mu.
            lower_candidates.append(float(np.max(-1.0 / mu10)))
        if mask11.any():
            # tau(1,1) = 1 - rho > 0  =>  rho < 1.
            upper_candidates.append(1.0)

        lo, hi = max(lower_candidates), min(upper_candidates)
        if lo >= hi:
            return 0.0
        center = 0.5 * (lo + hi)
        lo = center - (center - lo) * 0.999
        hi = center + (hi - center) * 0.999

        def neg_log_likelihood(rho: float) -> float:
            total = 0.0
            if mask00.any():
                total += np.sum(w00 * np.log(1.0 - lam00 * mu00 * rho))
            if mask01.any():
                total += np.sum(w01 * np.log(1.0 + lam01 * rho))
            if mask10.any():
                total += np.sum(w10 * np.log(1.0 + mu10 * rho))
            if mask11.any():
                total += np.sum(w11 * np.log(1.0 - rho))
            return -total

        result = optimize.minimize_scalar(
            neg_log_likelihood, bounds=(lo, hi), method="bounded"
        )
        if not getattr(result, "success", True):
            warnings.warn(
                f"Dixon-Coles rho fit did not converge: "
                f"{getattr(result, 'message', 'unknown reason')}; "
                "falling back to no correction (rho=0).",
                RuntimeWarning,
            )
            return 0.0
        return float(result.x)

    # -------------------------------------------------------------- predict
    def _warn_if_unseen(self, team: str) -> None:
        """Warn once per team about missing strengths (see __init__ note)."""
        if team not in self.attack and team not in self._warned_unseen:
            warnings.warn(
                f"Team '{team}' not seen in training data; using neutral "
                f"attack/defense strengths.",
                UserWarning,
            )
            self._warned_unseen.add(team)

    def expected_goals(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> Tuple[float, float]:
        """Return expected goals for (home, away)."""
        self._check_fitted("attack")
        self._warn_if_unseen(home_team)
        self._warn_if_unseen(away_team)
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
        """Joint probability matrix P[i, j] of home i goals, away j goals.

        Starts from independent Poisson margins, then applies the
        Dixon-Coles tau correction to the four low-score cells (see module
        docstring) when ``self.rho`` is non-zero. The four adjustments
        cancel exactly before renormalization, so total mass is preserved;
        ``predict_match`` and ``most_likely_score`` pick this up for free.
        """
        lam_home, lam_away = self.expected_goals(home_team, away_team, neutral)
        max_g = self.params.max_goals
        goals = np.arange(max_g + 1)
        p_home = poisson_dist.pmf(goals, lam_home)
        p_away = poisson_dist.pmf(goals, lam_away)
        matrix = np.outer(p_home, p_away)
        if self.params.dixon_coles and self.rho != 0.0:
            rho = self.rho
            matrix[0, 0] *= 1.0 - lam_home * lam_away * rho
            matrix[0, 1] *= 1.0 + lam_home * rho
            matrix[1, 0] *= 1.0 + lam_away * rho
            matrix[1, 1] *= 1.0 - rho
            matrix = np.clip(matrix, 0.0, None)  # numerical guard
        total = matrix.sum()
        if total > 0:
            matrix = matrix / total  # renormalize truncated tail (+ DC tweak)
        return matrix

    def _matrix_and_cumsum(
        self, home_team: str, away_team: str, neutral: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cache a fixture's scoreline matrix and its flattened cumsum.

        ``sample_score`` is called ~10^6 times per 10k-simulation run but
        only sees a bounded number of distinct (home, away, neutral) keys
        (<= teams^2 * 2), so caching here is what keeps simulation fast.
        """
        key = (home_team, away_team, neutral)
        if key not in self._matrix_cache:
            m = self.scoreline_matrix(home_team, away_team, neutral)
            self._matrix_cache[key] = (m, np.cumsum(m.ravel()))
        return self._matrix_cache[key]

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
        """Sample a single scoreline from the (Dixon-Coles corrected) joint.

        Draws via inverse-CDF over the cached scoreline matrix rather than
        two independent ``rng.poisson`` calls, so the Dixon-Coles
        correlation between the two teams' low scores is respected. Avoids
        ``rng.choice(p=...)`` (which rebuilds/validates the distribution on
        every call) since this runs ~10^6 times per 10k-simulation run.
        """
        matrix, cum = self._matrix_and_cumsum(home_team, away_team, neutral)
        idx = int(np.searchsorted(cum, rng.random()))
        idx = min(idx, cum.size - 1)  # guard against float rounding at cum[-1]
        return divmod(idx, matrix.shape[1])

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
        meta = pd.DataFrame(
            {
                "team": ["__intercept__", "__home_advantage__", "__rho__"],
                "attack": [self.intercept, self.home_advantage, self.rho],
                "defense": [0.0, 0.0, 0.0],
                "net_strength": [0.0, 0.0, 0.0],
            }
        )
        pd.concat([meta, table], ignore_index=True).to_csv(path, index=False)

    @classmethod
    def load(cls, path=None, config: Optional[Config] = None) -> "PoissonModel":
        """Load strengths from a CSV produced by :meth:`save`.

        Tolerates legacy CSVs that predate the ``__rho__`` meta row (loads
        with ``rho = 0.0``, i.e. plain independent Poisson), but requires the
        ``__intercept__`` / ``__home_advantage__`` meta rows -- without them
        the strengths cannot be turned into expected goals, so a CSV missing
        them (e.g. a hand-edited or non-``save()`` file) is rejected with a
        clear error rather than a bare ``KeyError``.
        """
        config = config or default_config()
        path = path or config.poisson_strengths_path
        df = pd.read_csv(path)
        model = cls(config)
        meta = df.set_index("team")
        required = ("__intercept__", "__home_advantage__")
        missing = [row for row in required if row not in meta.index]
        if missing:
            raise ValueError(
                f"{path} is missing required Poisson meta rows {missing}; "
                "it does not look like a file produced by PoissonModel.save()."
            )
        model.intercept = float(meta.loc["__intercept__", "attack"])
        model.home_advantage = float(meta.loc["__home_advantage__", "attack"])
        model.rho = (
            float(meta.loc["__rho__", "attack"]) if "__rho__" in meta.index else 0.0
        )
        teams_df = df[~df["team"].str.startswith("__")]
        model.attack = dict(zip(teams_df["team"], teams_df["attack"]))
        model.defense = dict(zip(teams_df["team"], teams_df["defense"]))
        model._matrix_cache.clear()
        return model
