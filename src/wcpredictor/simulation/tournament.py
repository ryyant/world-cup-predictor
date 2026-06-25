"""Monte Carlo simulation of the 48-team 2026 FIFA World Cup.

Format implemented:

* 12 groups (A-L) of 4 teams, single round-robin (6 matches each).
* The 12 group winners, 12 runners-up and the 8 best third-placed teams
  advance to a 32-team knockout bracket: Round of 32 -> Round of 16 ->
  Quarterfinals -> Semifinals -> Final.

The knockout bracket is built by seeding the 32 qualifiers (winners, then
runners-up, then best thirds, each ranked by group performance) into a standard
single-elimination bracket. This keeps stronger qualifiers apart in early
rounds. It is a deterministic, balanced approximation of FIFA's official
third-place slotting table (which depends on exactly which groups the eight
qualifying third-placed teams come from).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from wcpredictor.config import Config, default_config
from wcpredictor.simulation.match import simulate_match

HOST_TEAMS = frozenset({"USA", "Mexico", "Canada"})

# Knockout rounds, in order, naming the round that the *winners* of the
# previous round advance into. The full set of stages a team can "reach":
STAGES = (
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "winner",
)
KNOCKOUT_ROUND_NAMES = STAGES[1:]  # winners of R32 reach round_of_16, etc.


def _bracket_seed_order(n: int) -> List[int]:
    """Return seed numbers in standard bracket position order.

    For n=4 this is [1, 4, 3, 2] so that seeds 1 and 2 can only meet in the
    final. Pairs of adjacent entries are the first-round matchups.
    """
    seeds = [1]
    while len(seeds) < n:
        m = len(seeds) * 2
        nxt: List[int] = []
        for s in seeds:
            nxt.append(s)
            nxt.append(m + 1 - s)
        seeds = nxt
    return seeds


@dataclass
class TeamStats:
    played: int = 0
    win: int = 0
    draw: int = 0
    loss: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return 3 * self.win + self.draw

    @property
    def gd(self) -> int:
        return self.gf - self.ga


# Mutually exclusive "how far did they get" outcomes, group stage -> champion.
# Each maps to a display label; probabilities across these sum to 1 per team.
EXACT_OUTCOMES = (
    ("group_stage", "Group stage"),
    ("lost_r32", "Reached R32"),
    ("lost_r16", "Reached R16"),
    ("lost_qf", "Reached QF"),
    ("lost_sf", "Reached SF"),
    ("runner_up", "Runner-up"),
    ("champion", "Champion"),
)


@dataclass
class SimulationReport:
    """Aggregated probabilities across Monte Carlo runs."""

    table: pd.DataFrame
    n_simulations: int

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.table.head(n)

    def outcome_distribution(self) -> pd.DataFrame:
        """Mutually exclusive finish probabilities per team (sum to 1).

        Derived from the cumulative reach-stage probabilities: the chance a team
        is eliminated in the group stage, in each knockout round, finishes
        runner-up, or wins the title.
        """
        t = self.table
        out = pd.DataFrame({"team": t["team"], "group": t["group"]})
        out["group_stage"] = 1.0 - t["p_advance"]
        out["lost_r32"] = t["p_advance"] - t["p_round_of_16"]
        out["lost_r16"] = t["p_round_of_16"] - t["p_quarterfinal"]
        out["lost_qf"] = t["p_quarterfinal"] - t["p_semifinal"]
        out["lost_sf"] = t["p_semifinal"] - t["p_final"]
        out["runner_up"] = t["p_final"] - t["p_winner"]
        out["champion"] = t["p_winner"]
        # Clip tiny negative values from floating point subtraction.
        cols = [key for key, _ in EXACT_OUTCOMES]
        out[cols] = out[cols].clip(lower=0.0)
        return out

    def group_position_distribution(self) -> pd.DataFrame:
        """Probability of finishing 1st/2nd/3rd/4th in the group, per team."""
        cols = ["team", "group", "p_group_1st", "p_group_2nd",
                "p_group_3rd", "p_group_4th"]
        return self.table[cols].copy()


class TournamentSimulator:
    """Runs Monte Carlo simulations of the tournament."""

    def __init__(
        self,
        model,
        groups: pd.DataFrame,
        config: Optional[Config] = None,
    ):
        self.model = model
        self.config = config or default_config()
        self.sim_params = self.config.simulation
        self.groups: Dict[str, List[str]] = {
            g: list(sub.sort_values("slot")["team"])
            for g, sub in groups.groupby("group")
        }
        self.team_group = {
            team: g for g, teams in self.groups.items() for team in teams
        }

    # ------------------------------------------------------------ orientation
    def _orient(self, a: str, b: str) -> Tuple[str, str, bool]:
        """Decide home/away/neutral, giving host nations home advantage."""
        a_host, b_host = a in HOST_TEAMS, b in HOST_TEAMS
        if a_host and not b_host:
            return a, b, False
        if b_host and not a_host:
            return b, a, False
        return a, b, True

    def _play(self, a: str, b: str, rng, knockout: bool):
        home, away, neutral = self._orient(a, b)
        return simulate_match(
            self.model, home, away, rng, neutral=neutral, knockout=knockout
        )

    # --------------------------------------------------------------- group
    def simulate_group(self, teams: Sequence[str], rng) -> List[str]:
        """Play a group's round-robin and return teams ranked best-to-worst."""
        ranking, _stats = self._simulate_group_with_stats(teams, rng)
        return ranking

    def _criterion_value(self, team: str, stats: Dict[str, TeamStats], key: str):
        s = stats[team]
        if key == "points":
            return s.points
        if key == "goal_difference":
            return s.gd
        if key == "goals_for":
            return s.gf
        return 0

    def _rank(
        self,
        teams: List[str],
        stats: Dict[str, TeamStats],
        matches,
        rng,
    ) -> List[str]:
        """Rank teams applying configured tiebreakers, then random draw."""
        non_h2h = [
            k for k in self.sim_params.tiebreakers if k != "head_to_head"
        ]
        use_h2h = "head_to_head" in self.sim_params.tiebreakers

        def base_key(t):
            return tuple(
                self._criterion_value(t, stats, k) for k in non_h2h
            )

        teams_sorted = sorted(teams, key=base_key, reverse=True)

        # Resolve blocks tied on all non-head-to-head criteria.
        ordered: List[str] = []
        i = 0
        while i < len(teams_sorted):
            j = i + 1
            while j < len(teams_sorted) and base_key(
                teams_sorted[j]
            ) == base_key(teams_sorted[i]):
                j += 1
            block = teams_sorted[i:j]
            if len(block) > 1:
                block = self._break_tie(block, matches, rng, use_h2h)
            ordered.extend(block)
            i = j
        return ordered

    def _break_tie(self, block, matches, rng, use_h2h) -> List[str]:
        block_set = set(block)
        mini = {t: TeamStats() for t in block}
        if use_h2h:
            for a, b, ag, bg in matches:
                if a in block_set and b in block_set:
                    sa, sb = mini[a], mini[b]
                    sa.gf += ag
                    sa.ga += bg
                    sb.gf += bg
                    sb.ga += ag
                    if ag > bg:
                        sa.win += 1
                        sb.loss += 1
                    elif bg > ag:
                        sb.win += 1
                        sa.loss += 1
                    else:
                        sa.draw += 1
                        sb.draw += 1
        # Random final tiebreak (drawing of lots), deterministic per call.
        noise = {t: rng.random() for t in block}
        return sorted(
            block,
            key=lambda t: (
                mini[t].points,
                mini[t].gd,
                mini[t].gf,
                noise[t],
            ),
            reverse=True,
        )

    def _rank_cross_group(self, teams_with_stats, rng) -> List[str]:
        """Rank teams from different groups by (points, gd, gf) then random."""
        noise = {t: rng.random() for t, _ in teams_with_stats}
        return [
            t
            for t, _ in sorted(
                teams_with_stats,
                key=lambda item: (
                    item[1].points,
                    item[1].gd,
                    item[1].gf,
                    noise[item[0]],
                ),
                reverse=True,
            )
        ]

    # ------------------------------------------------------- single tournament
    def simulate_once(self, rng) -> Dict[str, object]:
        """Simulate one full tournament.

        Returns a dict with the group standings, the 32 qualifiers, the set of
        teams reaching each knockout stage, and the champion. Useful for
        notebook demos and bracket printouts.
        """
        winners: List[Tuple[str, TeamStats]] = []
        runners: List[Tuple[str, TeamStats]] = []
        thirds: List[Tuple[str, TeamStats]] = []
        standings: Dict[str, List[str]] = {}

        for g, teams in self.groups.items():
            ranking, stats = self._simulate_group_with_stats(teams, rng)
            standings[g] = ranking
            winners.append((ranking[0], stats[ranking[0]]))
            runners.append((ranking[1], stats[ranking[1]]))
            thirds.append((ranking[2], stats[ranking[2]]))

        best_thirds = self._rank_cross_group(thirds, rng)[:8]
        qualifiers = (
            self._rank_cross_group(winners, rng)
            + self._rank_cross_group(runners, rng)
            + best_thirds
        )

        reached: Dict[str, set] = {s: set() for s in STAGES}
        reached["round_of_32"] = set(qualifiers)

        seed_order = _bracket_seed_order(32)
        current = [qualifiers[s - 1] for s in seed_order]
        for stage_name in KNOCKOUT_ROUND_NAMES:
            next_round: List[str] = []
            for k in range(0, len(current), 2):
                res = self._play(
                    current[k], current[k + 1], rng, knockout=True
                )
                next_round.append(res.winner)
            reached[stage_name] = set(next_round)
            current = next_round

        return {
            "standings": standings,
            "qualifiers": qualifiers,
            "reached": reached,
            "champion": current[0],
        }

    # ------------------------------------------------------------------- run
    def run(self, n_simulations: Optional[int] = None) -> SimulationReport:
        n = n_simulations or self.sim_params.n_simulations
        rng = np.random.default_rng(self.sim_params.random_seed)

        all_teams = list(self.team_group.keys())
        counts = {
            stage: {t: 0 for t in all_teams} for stage in STAGES
        }
        # Group finishing position counts (1st..4th) per team.
        position_counts = {t: [0, 0, 0, 0] for t in all_teams}

        seed_order = _bracket_seed_order(32)

        for _ in range(n):
            winners: List[Tuple[str, TeamStats]] = []
            runners: List[Tuple[str, TeamStats]] = []
            thirds: List[Tuple[str, TeamStats]] = []

            for teams in self.groups.values():
                ranking, stats = self._simulate_group_with_stats(teams, rng)
                for pos, team in enumerate(ranking):
                    position_counts[team][pos] += 1
                winners.append((ranking[0], stats[ranking[0]]))
                runners.append((ranking[1], stats[ranking[1]]))
                thirds.append((ranking[2], stats[ranking[2]]))

            best_thirds = self._rank_cross_group(thirds, rng)[:8]

            ranked_winners = self._rank_cross_group(winners, rng)
            ranked_runners = self._rank_cross_group(runners, rng)
            qualifiers = ranked_winners + ranked_runners + best_thirds  # 32

            for t in qualifiers:
                counts["round_of_32"][t] += 1

            # Seed into the bracket. Seed 1 = best qualifier.
            bracket = [qualifiers[s - 1] for s in seed_order]
            current = bracket
            for stage_name in KNOCKOUT_ROUND_NAMES:
                next_round: List[str] = []
                for k in range(0, len(current), 2):
                    res = self._play(
                        current[k], current[k + 1], rng, knockout=True
                    )
                    next_round.append(res.winner)
                for t in next_round:
                    counts[stage_name][t] += 1
                current = next_round

        return self._build_report(counts, position_counts, n, all_teams)

    def _simulate_group_with_stats(
        self, teams: Sequence[str], rng
    ) -> Tuple[List[str], Dict[str, TeamStats]]:
        stats = {t: TeamStats() for t in teams}
        played_matches = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                res = self._play(a, b, rng, knockout=False)
                if res.home_team == a:
                    ag, bg = res.home_score, res.away_score
                else:
                    ag, bg = res.away_score, res.home_score
                played_matches.append((a, b, ag, bg))
                sa, sb = stats[a], stats[b]
                sa.played += 1
                sb.played += 1
                sa.gf += ag
                sa.ga += bg
                sb.gf += bg
                sb.ga += ag
                if ag > bg:
                    sa.win += 1
                    sb.loss += 1
                elif bg > ag:
                    sb.win += 1
                    sa.loss += 1
                else:
                    sa.draw += 1
                    sb.draw += 1
        ranking = self._rank(list(teams), stats, played_matches, rng)
        return ranking, stats

    def _build_report(
        self, counts, position_counts, n, all_teams
    ) -> SimulationReport:
        rows = []
        for t in all_teams:
            row = {"team": t, "group": self.team_group[t]}
            for stage in STAGES:
                row[f"p_{stage}"] = counts[stage][t] / n
            pos = position_counts[t]
            row["p_group_1st"] = pos[0] / n
            row["p_group_2nd"] = pos[1] / n
            row["p_group_3rd"] = pos[2] / n
            row["p_group_4th"] = pos[3] / n
            rows.append(row)
        table = pd.DataFrame(rows)
        table = table.rename(columns={"p_round_of_32": "p_advance"})
        table = table.sort_values(
            ["p_winner", "p_final", "p_advance"], ascending=False
        ).reset_index(drop=True)
        return SimulationReport(table=table, n_simulations=n)
