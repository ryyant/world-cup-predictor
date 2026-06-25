"""Advanced plotting helpers for predictions and simulation outcomes.

These functions turn the models and :class:`SimulationReport` into publication-
quality matplotlib figures. They are intentionally kept out of the package's
top-level ``__init__`` so importing :mod:`wcpredictor` (and the CLI) stays light;
import them explicitly:

    from wcpredictor.visualization import plot_outcome_distribution

Every function accepts an optional ``ax`` (or creates its own figure) and
returns the matplotlib ``Axes`` so plots compose nicely in notebooks.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

from wcpredictor.simulation.tournament import EXACT_OUTCOMES

# Colour per mutually-exclusive finish: light -> blues for the rounds, then
# medal colours (bronze / silver / gold) for the deepest runs.
OUTCOME_COLORS = {
    "group_stage": "#ecf0f1",
    "lost_r32": "#d4e6f1",
    "lost_r16": "#a9cce3",
    "lost_qf": "#7fb3d5",
    "lost_sf": "#cd7f32",   # bronze (semifinal exit)
    "runner_up": "#bdc3c7",  # silver
    "champion": "#f1c40f",   # gold
}

STAGE_COLUMNS = [
    ("p_advance", "Advance\n(R32)"),
    ("p_round_of_16", "R16"),
    ("p_quarterfinal", "QF"),
    ("p_semifinal", "SF"),
    ("p_final", "Final"),
    ("p_winner", "Champion"),
]

GROUP_POSITION_COLUMNS = [
    ("p_group_1st", "1st", "#1a9850"),
    ("p_group_2nd", "2nd", "#91cf60"),
    ("p_group_3rd", "3rd", "#fee08b"),
    ("p_group_4th", "4th", "#d73027"),
]

_DEEP_RUN_KEYS = ["champion", "runner_up", "lost_sf", "lost_qf",
                  "lost_r16", "lost_r32"]


def _percent(x: float) -> str:
    return f"{x * 100:.0f}%"


def plot_outcome_distribution(
    report,
    top_n: int = 16,
    ax: Optional[plt.Axes] = None,
    min_label: float = 0.04,
) -> plt.Axes:
    """Stacked horizontal bars of each team's full finish distribution.

    Every bar spans 0-100% and is split into mutually exclusive outcomes from
    "group stage" (left) to "champion" (right), so you can read a team's entire
    range of plausible results at a glance.
    """
    dist = report.outcome_distribution()
    dist = dist.sort_values(_DEEP_RUN_KEYS, ascending=False).head(top_n)
    dist = dist.iloc[::-1].reset_index(drop=True)  # best at top

    if ax is None:
        _fig, ax = plt.subplots(figsize=(11, 0.45 * len(dist) + 1.5))

    y = np.arange(len(dist))
    left = np.zeros(len(dist))
    for key, label in EXACT_OUTCOMES:
        widths = dist[key].to_numpy()
        ax.barh(y, widths, left=left, color=OUTCOME_COLORS[key],
                edgecolor="white", linewidth=0.6, label=label)
        for yi, (w, l) in enumerate(zip(widths, left)):
            if w >= min_label:
                ax.text(l + w / 2, yi, _percent(w), ha="center", va="center",
                        fontsize=8,
                        color="#2c3e50" if key != "lost_sf" else "white")
        left += widths

    ax.set_yticks(y)
    ax.set_yticklabels(dist["team"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(f"How far will they go? Finish distribution (top {len(dist)})")
    ax.xaxis.set_major_formatter(lambda v, _pos: _percent(v))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(EXACT_OUTCOMES), frameon=False, fontsize=8)
    ax.margins(y=0.01)
    return ax


def plot_stage_heatmap(
    report,
    top_n: int = 20,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrRd",
) -> plt.Axes:
    """Heatmap of cumulative reach-stage probabilities (teams x stages)."""
    table = report.table.head(top_n)
    cols = [c for c, _ in STAGE_COLUMNS]
    labels = [l for _, l in STAGE_COLUMNS]
    data = table[cols].to_numpy()

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 0.42 * len(table) + 1.5))

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["team"])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, _percent(val), ha="center", va="center", fontsize=8,
                    color="white" if val > 0.55 else "#2c3e50")
    ax.set_title(f"Probability of reaching each stage (top {len(table)})")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("probability", rotation=90)
    return ax


def plot_title_race(
    report,
    top_n: int = 12,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Lollipop chart of title (championship) probabilities."""
    table = report.table.head(top_n).iloc[::-1]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(9, 0.4 * len(table) + 1.5))

    y = np.arange(len(table))
    probs = table["p_winner"].to_numpy()
    cmap = colormaps.get_cmap("autumn_r")
    colors = cmap(np.linspace(0.25, 0.95, len(table)))
    ax.hlines(y, 0, probs, color="#cccccc", linewidth=1.5, zorder=1)
    ax.scatter(probs, y, s=140, color=colors, zorder=2, edgecolor="white")
    for yi, p in zip(y, probs):
        ax.text(p, yi, "  " + _percent(p), va="center", ha="left", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(table["team"])
    ax.set_xlabel("P(win the World Cup)")
    ax.set_xlim(0, max(probs.max() * 1.18, 0.05))
    ax.xaxis.set_major_formatter(lambda v, _pos: _percent(v))
    ax.set_title(f"Title race (top {len(table)})")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_group_outcomes(
    report,
    group: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Stacked bars of finishing-position probabilities for one group."""
    gp = report.group_position_distribution()
    sub = gp[gp["group"] == group].copy()
    # Order teams by their chance of topping the group.
    sub = sub.sort_values("p_group_1st", ascending=False)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    for col, label, color in GROUP_POSITION_COLUMNS:
        vals = sub[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 0.06:
                ax.text(xi, b + v / 2, _percent(v), ha="center", va="center",
                        fontsize=8, color="#2c3e50")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sub["team"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.yaxis.set_major_formatter(lambda v, _pos: _percent(v))
    ax.set_title(f"Group {group}: finishing position")
    ax.legend(title="Finish", loc="upper right", fontsize=8, ncol=4,
              bbox_to_anchor=(1.0, 1.16), frameon=False)
    return ax


def plot_group_grid(report, ncols: int = 4) -> plt.Figure:
    """Grid of group-outcome charts for every group in the draw."""
    groups = sorted(report.table["group"].unique())
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, group in zip(axes, groups):
        plot_group_outcomes(report, group, ax=ax)
        ax.legend().set_visible(False)
        ax.set_xlabel("")
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Group-stage finishing positions", y=1.0, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.legend(handles, labels, title="Group finish", loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    return fig


def plot_match_comparison(
    predictions: Dict[str, "object"],
    home_team: str,
    away_team: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Grouped bars comparing W/D/L probabilities across models.

    ``predictions`` maps a model name to a ``MatchPrediction``.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 4.5))

    model_names = list(predictions)
    outcomes = [f"{home_team} win", "Draw", f"{away_team} win"]
    colors = ["#2980b9", "#95a5a6", "#c0392b"]
    x = np.arange(len(outcomes))
    width = 0.8 / max(len(model_names), 1)

    for i, name in enumerate(model_names):
        pred = predictions[name]
        vals = [pred.p_home_win, pred.p_draw, pred.p_away_win]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=colors, alpha=0.55 + 0.45 * i / max(len(model_names) - 1, 1),
                      edgecolor="white")
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01,
                    _percent(v), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.yaxis.set_major_formatter(lambda v, _pos: _percent(v))
    ax.set_title(f"{home_team} vs {away_team}: outcome by model")
    ax.legend(title="Model", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_scoreline_heatmap(
    model,
    home_team: str,
    away_team: str,
    neutral: bool = True,
    max_goals: int = 6,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Annotated scoreline-probability heatmap for a fixture (Poisson model).

    Cells are tinted by who the scoreline favours (home win / draw / away win),
    with the most likely scoreline highlighted.
    """
    matrix = model.scoreline_matrix(home_team, away_team, neutral)
    matrix = matrix[: max_goals + 1, : max_goals + 1]

    # Build an RGBA image: blue-ish for home wins, grey for draws, red for away.
    home_cmap = LinearSegmentedColormap.from_list(
        "home", ["#ffffff", "#2980b9"])
    draw_cmap = LinearSegmentedColormap.from_list(
        "draw", ["#ffffff", "#7f8c8d"])
    away_cmap = LinearSegmentedColormap.from_list(
        "away", ["#ffffff", "#c0392b"])

    norm = matrix / matrix.max() if matrix.max() > 0 else matrix
    rgba = np.zeros(matrix.shape + (4,))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            shade = norm[i, j]
            if i > j:
                rgba[i, j] = home_cmap(shade)
            elif i == j:
                rgba[i, j] = draw_cmap(shade)
            else:
                rgba[i, j] = away_cmap(shade)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.imshow(rgba, origin="lower", aspect="equal")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j] * 100:.0f}", ha="center", va="center",
                    fontsize=7, color="#2c3e50")

    # Highlight the most likely scoreline.
    mi, mj = np.unravel_index(np.argmax(matrix), matrix.shape)
    ax.add_patch(plt.Rectangle((mj - 0.5, mi - 0.5), 1, 1, fill=False,
                               edgecolor="#f1c40f", linewidth=3))

    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    ax.set_xlabel(f"{away_team} goals")
    ax.set_ylabel(f"{home_team} goals")
    ax.set_xticks(range(max_goals + 1))
    ax.set_yticks(range(max_goals + 1))
    ax.set_title(
        f"Scoreline probabilities (%)\n"
        f"{home_team} {_percent(p_home)}  |  Draw {_percent(p_draw)}  |  "
        f"{away_team} {_percent(p_away)}"
    )
    return ax
