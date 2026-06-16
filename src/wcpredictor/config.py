"""Central configuration: filesystem paths and model hyperparameters.

All tunable knobs live here so the rest of the package stays free of magic
numbers. Use :func:`default_config` for the standard setup, or build a custom
:class:`Config` for experiments (e.g. in notebooks).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Tuple

# Project layout. config.py lives at src/wcpredictor/config.py, so the project
# root is three parents up.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# Vendored upstream tournament files (openfootball/worldcup.json); the 2026
# file also carries already-played fixtures used to condition the simulation.
SOURCE_DATA_DIR = DATA_DIR / "source"

# Seed CSV file names (bundled under data/raw).
MATCHES_CSV = "matches.csv"
TEAMS_CSV = "teams.csv"
GROUPS_CSV = "wc2026_groups.csv"

# Vendored 2026 fixtures/results (under data/source), the source of truth for
# which matches have already been played when conditioning the simulation.
WC2026_SOURCE = "2026.worldcup.json"

# Saved artifact file names (written under data/processed).
ELO_RATINGS_FILE = "elo_ratings.csv"
POISSON_STRENGTHS_FILE = "poisson_strengths.csv"


@dataclass(frozen=True)
class EloParams:
    """Hyperparameters for the Elo rating model."""

    initial_rating: float = 1500.0
    k_factor: float = 40.0
    home_advantage: float = 65.0
    # Logistic scale: a `scale`-point edge maps to ~10x odds.
    scale: float = 400.0
    # Margin-of-victory multiplier dampening (FIFA-style). 0 disables MOV.
    mov_factor: float = 1.0
    # Shape of the FIFA-style MOV dampening: a larger `mov_dampening` softens
    # the log-margin growth, and `mov_rating_slope` shrinks the multiplier as
    # the rating gap widens (favourites winning big count less).
    mov_dampening: float = 2.2
    mov_rating_slope: float = 0.001
    # Draw model: probability of a draw at equal strength (used to split the
    # Elo expected score into win/draw/loss).
    draw_base: float = 0.27
    # Clip range for a data-driven draw base fitted in a later task; keep
    # `draw_base` above as the fallback/prior.
    fitted_draw_base_bounds: Tuple[float, float] = (0.05, 0.45)


@dataclass(frozen=True)
class PoissonParams:
    """Hyperparameters for the Poisson goals model."""

    # Baseline average goals per team per match (league-wide intercept seed).
    base_goals: float = 1.35
    # Home advantage expressed as a multiplicative factor on expected goals.
    home_factor: float = 1.15
    # Recency weighting half-life in days for fitting strengths (0 disables).
    # With finals-only data (~984 matches over 96 years, one tournament every 4
    # years), a 5-year half-life leaves an effective sample of only ~108 matches
    # vs ~170 Poisson parameters; 10 years keeps the last 2-3 tournaments
    # dominant while older cups still contribute (~214 effective matches).
    half_life_days: float = 3650.0
    # Maximum goals considered when building the scoreline distribution.
    max_goals: int = 10
    # Ridge-style shrinkage of attack/defense strengths toward the mean.
    shrinkage: float = 0.30
    # Dixon-Coles low-score correlation correction (wired in a later task).
    dixon_coles: bool = True
    # Clip range for the Dixon-Coles rho parameter fitted in a later task.
    rho_bounds: Tuple[float, float] = (-0.2, 0.2)


@dataclass(frozen=True)
class SimulationParams:
    """Hyperparameters for the Monte Carlo tournament simulation."""

    n_simulations: int = 10000
    random_seed: int = 42
    # Weight of the stronger side's edge in a penalty shootout: 0 is a coin
    # flip, 1 uses the full attacking edge. 0.5 halves the edge.
    penalty_edge_weight: float = 0.5
    # Group-stage tiebreaker order applied sequentially.
    # Supported keys: points, goal_difference, goals_for, head_to_head.
    tiebreakers: Tuple[str, ...] = (
        "points",
        "goal_difference",
        "goals_for",
        "head_to_head",
    )


@dataclass(frozen=True)
class Config:
    """Top-level configuration bundle."""

    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    source_data_dir: Path = SOURCE_DATA_DIR
    matches_csv: str = MATCHES_CSV
    teams_csv: str = TEAMS_CSV
    groups_csv: str = GROUPS_CSV
    wc2026_source: str = WC2026_SOURCE
    elo: EloParams = field(default_factory=EloParams)
    poisson: PoissonParams = field(default_factory=PoissonParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)

    @property
    def matches_path(self) -> Path:
        return self.raw_data_dir / self.matches_csv

    @property
    def teams_path(self) -> Path:
        return self.raw_data_dir / self.teams_csv

    @property
    def groups_path(self) -> Path:
        return self.raw_data_dir / self.groups_csv

    @property
    def wc2026_source_path(self) -> Path:
        return self.source_data_dir / self.wc2026_source

    @property
    def elo_ratings_path(self) -> Path:
        return self.processed_data_dir / ELO_RATINGS_FILE

    @property
    def poisson_strengths_path(self) -> Path:
        return self.processed_data_dir / POISSON_STRENGTHS_FILE

    def with_overrides(self, **kwargs) -> "Config":
        """Return a copy of this config with top-level fields replaced."""
        return replace(self, **kwargs)


def default_config() -> Config:
    """Return the standard configuration used by the CLI and notebooks."""
    return Config()
