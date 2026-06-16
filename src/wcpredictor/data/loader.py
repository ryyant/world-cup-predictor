"""Load and validate the bundled CSV datasets into pandas DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from wcpredictor.config import Config, default_config

MATCH_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral",
    "tournament",
}
TEAM_COLUMNS = {"team", "confederation", "pot"}
GROUP_COLUMNS = {"group", "slot", "team"}


class DataValidationError(ValueError):
    """Raised when a CSV is missing columns or contains invalid rows."""


def _require_columns(df: pd.DataFrame, required: set, source: Path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(
            f"{source.name} is missing required columns: {sorted(missing)}"
        )


def load_matches(config: Optional[Config] = None) -> pd.DataFrame:
    """Load historical match results.

    Returns a DataFrame with parsed ``date`` (datetime64), integer scores, a
    boolean ``neutral`` flag, sorted chronologically.
    """
    config = config or default_config()
    path = config.matches_path
    if not path.exists():
        raise FileNotFoundError(
            f"Matches file not found at {path}. Run scripts/generate_seed_data.py "
            "or place your own matches.csv there."
        )
    df = pd.read_csv(path)
    _require_columns(df, MATCH_COLUMNS, path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise DataValidationError(f"{path.name} contains unparseable dates.")

    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or (df[col] < 0).any():
            raise DataValidationError(
                f"{path.name} has missing or negative values in '{col}'."
            )
        df[col] = df[col].astype(int)

    df["neutral"] = df["neutral"].astype(bool)
    for col in ("home_team", "away_team"):
        df[col] = df[col].astype(str).str.strip()
    if (df["home_team"] == df["away_team"]).any():
        raise DataValidationError(
            f"{path.name} has rows where a team plays itself."
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_teams(config: Optional[Config] = None) -> pd.DataFrame:
    """Load team metadata (confederation, seeding pot)."""
    config = config or default_config()
    path = config.teams_path
    if not path.exists():
        raise FileNotFoundError(f"Teams file not found at {path}.")
    df = pd.read_csv(path)
    _require_columns(df, TEAM_COLUMNS, path)
    df["team"] = df["team"].astype(str).str.strip()
    if df["team"].duplicated().any():
        dupes = df.loc[df["team"].duplicated(), "team"].tolist()
        raise DataValidationError(f"{path.name} has duplicate teams: {dupes}")
    df["pot"] = df["pot"].astype(int)
    return df.reset_index(drop=True)


def load_groups(config: Optional[Config] = None) -> pd.DataFrame:
    """Load the 2026 group draw (12 groups of 4)."""
    config = config or default_config()
    path = config.groups_path
    if not path.exists():
        raise FileNotFoundError(f"Groups file not found at {path}.")
    df = pd.read_csv(path)
    _require_columns(df, GROUP_COLUMNS, path)
    df["group"] = df["group"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.strip()
    df["slot"] = df["slot"].astype(int)

    sizes = df.groupby("group").size()
    bad = sizes[sizes != 4]
    if not bad.empty:
        raise DataValidationError(
            f"{path.name}: every group must have 4 teams; offending: "
            f"{bad.to_dict()}"
        )
    if df["team"].duplicated().any():
        dupes = df.loc[df["team"].duplicated(), "team"].tolist()
        raise DataValidationError(
            f"{path.name} has a team in more than one group: {dupes}"
        )
    return df.sort_values(["group", "slot"]).reset_index(drop=True)
