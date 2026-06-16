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
_STAGE_INDEX = {stage: i for i, stage in enumerate(STAGES)}


def _bracket_seed_order(n: int) -> List[int]:
    """Return seed numbers in standard bracket position order.

    For n=4 this is [1, 4, 2, 3] so that seeds 1 and 2 land in opposite
    halves and can only meet in the final. Pairs of adjacent entries are the
    first-round matchups.
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
    win: int = 0
    draw: int = 0
    loss: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def played(self) -> int:
        return self.win + self.draw + self.loss

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

    def standard_errors(self) -> pd.DataFrame:
        """Binomial standard error sqrt(p(1-p)/n) for every probability column.

        Each ``p_*`` column in ``table`` is a Monte Carlo estimate of a true
        probability from ``n_simulations`` Bernoulli trials (did the team
        reach that stage in this simulated tournament, yes/no), so its
        sampling error is exactly the binomial standard error. Returns one
        ``se_<name>`` column per ``p_<name>`` column (e.g. ``p_winner`` ->
        ``se_winner``), alongside the same identifying columns (team, group)
        as ``table``. ``self.table`` itself is left unchanged.
        """
        id_cols = [c for c in self.table.columns if not c.startswith("p_")]
        p_cols = [c for c in self.table.columns if c.startswith("p_")]
        out = self.table[id_cols].copy()
        for col in p_cols:
            p = self.table[col]
            out["se_" + col[len("p_"):]] = np.sqrt(
                p * (1.0 - p) / self.n_simulations
            )
        return out


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
        # The bracket assumes 12 winners + 12 runners-up + 8 best thirds = 32
        # qualifiers. Guard here too (not just in load_groups) since a groups
        # DataFrame can be built directly, e.g. in tests/notebooks -- a wrong
        # count would otherwise surface as an opaque IndexError in the bracket.
        if len(self.groups) != 12:
            raise ValueError(
                "TournamentSimulator needs exactly 12 groups (got "
                f"{len(self.groups)}); the knockout bracket seeds 32 qualifiers."
            )
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
            self.model, home, away, rng, neutral=neutral, knockout=knockout,
            penalty_edge_weight=self.sim_params.penalty_edge_weight,
        )

    # ------------------------------------------------------------- knockout
    def _play_knockout(self, qualifiers: Sequence[str], rng) -> Dict[str, List[str]]:
        """Seed the 32 qualifiers into the bracket and play all knockout rounds.

        Returns {stage_name: [teams reaching that stage]} for each stage in
        KNOCKOUT_ROUND_NAMES; the final stage's list has one element (the champion).
        """
        seed_order = _bracket_seed_order(32)
        current = [qualifiers[s - 1] for s in seed_order]
        rounds: Dict[str, List[str]] = {}
        for stage_name in KNOCKOUT_ROUND_NAMES:
            next_round: List[str] = []
            for k in range(0, len(current), 2):
                res = self._play(current[k], current[k + 1], rng, knockout=True)
                next_round.append(res.winner)
            rounds[stage_name] = next_round
            current = next_round
        return rounds

    def _play_frontier(self, frontier, remaining_stages, rng) -> Dict[str, List[str]]:
        """Play forward from a real, in-progress knockout round.

        ``frontier`` is the current round's ties in bracket order (objects with
        ``team1``/``team2`` and an optional already-decided ``winner``);
        ``remaining_stages`` names the stages their winners go on to reach,
        ending with ``"winner"``. Returns ``{stage: [teams reaching it]}`` for
        every stage in ``remaining_stages``.
        """
        current: List[str] = []
        for tie in frontier:
            winner = tie.winner
            if winner is None:
                winner = self._play(tie.team1, tie.team2, rng, knockout=True).winner
            current.append(winner)

        rounds: Dict[str, List[str]] = {}
        for i, stage_name in enumerate(remaining_stages):
            # ``current`` holds exactly the teams that have reached this stage.
            rounds[stage_name] = current
            if i == len(remaining_stages) - 1:
                break  # the final stage ("winner") is a single team, not replayed
            next_round: List[str] = []
            for k in range(0, len(current), 2):
                res = self._play(current[k], current[k + 1], rng, knockout=True)
                next_round.append(res.winner)
            current = next_round
        return rounds

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
    def _simulate_group_stage(
        self, rng
    ) -> Tuple[
        List[Tuple[str, TeamStats]],
        List[Tuple[str, TeamStats]],
        List[Tuple[str, TeamStats]],
        Dict[str, List[str]],
    ]:
        """Play every group's round-robin.

        Returns ``(winners, runners, thirds, standings)`` where the first three
        are lists of ``(team, TeamStats)`` for the group winners, runners-up
        and third-placed teams, and ``standings`` maps each group to its full
        best-to-worst ranking.
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
        return winners, runners, thirds, standings

    def _qualifiers_from_group_stage(
        self, winners, runners, thirds, rng
    ) -> List[str]:
        """Rank winners / runners-up / best-8 thirds into 32 seeded qualifiers.

        The cross-group rankings are drawn in the order thirds, winners,
        runners so the random tiebreaks consume ``rng`` identically for both
        :meth:`simulate_once` and :meth:`run`.
        """
        best_thirds = self._rank_cross_group(thirds, rng)[:8]
        ranked_winners = self._rank_cross_group(winners, rng)
        ranked_runners = self._rank_cross_group(runners, rng)
        return ranked_winners + ranked_runners + best_thirds

    def simulate_once(self, rng) -> Dict[str, object]:
        """Simulate one full tournament.

        Returns a dict with the group standings, the 32 qualifiers, the set of
        teams reaching each knockout stage, and the champion. Useful for
        notebook demos and bracket printouts.
        """
        winners, runners, thirds, standings = self._simulate_group_stage(rng)
        qualifiers = self._qualifiers_from_group_stage(
            winners, runners, thirds, rng
        )

        rounds = self._play_knockout(qualifiers, rng)
        reached: Dict[str, set] = {s: set() for s in STAGES}
        reached["round_of_32"] = set(qualifiers)
        for stage_name, survivors in rounds.items():
            reached[stage_name] = set(survivors)

        return {
            "standings": standings,
            "qualifiers": qualifiers,
            "reached": reached,
            "champion": rounds[KNOCKOUT_ROUND_NAMES[-1]][0],
        }

    # ------------------------------------------------------------------- run
    def run(self, n_simulations: Optional[int] = None) -> SimulationReport:
        n = n_simulations if n_simulations is not None else self.sim_params.n_simulations
        if n <= 0:
            raise ValueError(f"n_simulations must be positive, got {n}.")
        rng = np.random.default_rng(self.sim_params.random_seed)

        all_teams = list(self.team_group.keys())
        counts = {
            stage: {t: 0 for t in all_teams} for stage in STAGES
        }
        # Group finishing position counts (1st..4th) per team.
        position_counts = {t: [0, 0, 0, 0] for t in all_teams}

        for _ in range(n):
            winners, runners, thirds, standings = self._simulate_group_stage(rng)
            for ranking in standings.values():
                for pos, team in enumerate(ranking):
                    position_counts[team][pos] += 1

            qualifiers = self._qualifiers_from_group_stage(
                winners, runners, thirds, rng
            )
            for t in qualifiers:
                counts["round_of_32"][t] += 1

            rounds = self._play_knockout(qualifiers, rng)
            for stage_name, survivors in rounds.items():
                for t in survivors:
                    counts[stage_name][t] += 1

        return self._build_report(counts, position_counts, n, all_teams)

    # ----------------------------------------------------- conditioned run
    def run_conditioned(
        self, state, n_simulations: Optional[int] = None
    ) -> SimulationReport:
        """Project the tournament *given the results already played*.

        Unlike :meth:`run`, which replays the whole event from scratch, this
        locks in ``state`` (from
        :func:`wcpredictor.data.tournament_state.load_tournament_state`): the
        group stage and the completed knockout rounds are treated as settled,
        so a team's probability of reaching any already-decided stage is exactly
        0 or 1, and only the frontier round onward is simulated. Eliminated
        teams therefore get a 0% chance of winning -- the whole point of
        conditioning once the tournament is under way.

        Returns a :class:`SimulationReport` with the same columns as
        :meth:`run`, so every downstream visualization works unchanged.
        """
        n = n_simulations if n_simulations is not None else self.sim_params.n_simulations
        if n <= 0:
            raise ValueError(f"n_simulations must be positive, got {n}.")
        rng = np.random.default_rng(self.sim_params.random_seed)

        all_teams = list(self.team_group.keys())
        counts = {stage: {t: 0 for t in all_teams} for stage in STAGES}
        frontier_idx = _STAGE_INDEX[state.frontier_stage]

        # Settled prefix: a team "reached" every stage up to and including the
        # frontier that its actual run got to -- deterministic 0/1 (== n / n).
        for t in all_teams:
            reached_idx = state.reached_index(t)
            for stage in STAGES:
                if _STAGE_INDEX[stage] <= frontier_idx and reached_idx >= _STAGE_INDEX[stage]:
                    counts[stage][t] = n

        # Uncertain suffix: roll the frontier forward n times.
        remaining = state.remaining_stages
        for _ in range(n):
            rounds = self._play_frontier(state.frontier, remaining, rng)
            for stage, survivors in rounds.items():
                for t in survivors:
                    counts[stage][t] += 1

        # Group finishing positions are settled facts now, not probabilities.
        position_counts = {t: [0, 0, 0, 0] for t in all_teams}
        for t in all_teams:
            pos = state.group_position.get(t)
            if pos in (1, 2, 3, 4):
                position_counts[t][pos - 1] = n

        return self._build_report(counts, position_counts, n, all_teams)

    def simulate_once_conditioned(self, state, rng) -> Dict[str, object]:
        """One realization of the *remaining* tournament, given ``state``.

        Mirrors :meth:`simulate_once` but rolls dice only from the frontier
        round onward; the ``reached`` sets fold in the teams that have already
        locked those stages in.
        """
        rounds = self._play_frontier(state.frontier, state.remaining_stages, rng)
        reached: Dict[str, set] = {s: set() for s in STAGES}
        frontier_idx = _STAGE_INDEX[state.frontier_stage]
        for t in self.team_group:
            reached_idx = state.reached_index(t)
            for stage in STAGES:
                if _STAGE_INDEX[stage] <= frontier_idx and reached_idx >= _STAGE_INDEX[stage]:
                    reached[stage].add(t)
        for stage, survivors in rounds.items():
            reached[stage].update(survivors)
        return {
            "reached": reached,
            "champion": rounds[state.remaining_stages[-1]][0],
        }

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
