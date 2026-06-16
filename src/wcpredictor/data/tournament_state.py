"""Parse the actual state of the 2026 World Cup from the vendored fixtures.

The simulation in :mod:`wcpredictor.simulation.tournament` can replay the whole
event from scratch (useful *before* a ball is kicked). Once the tournament is
under way, though, most of the bracket is no longer uncertain -- the group
stage and the early knockout rounds have real results, and teams that lost are
out. Re-simulating those settled rounds gives already-eliminated sides (Germany,
Netherlands, ...) a championship probability they can no longer have.

This module reads ``data/source/2026.worldcup.json`` and distils it into a
:class:`TournamentState`: how far every team has already gone, their final group
positions, and the *frontier* -- the earliest knockout round that still has
matches left to play, with its real matchups. The simulator conditions on that
state and only rolls dice for what is genuinely still undecided.

Only the knockout phase is conditioned incrementally; a partially played group
stage is not supported (it raises), because the 2026 group stage is complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wcpredictor.config import Config, default_config
from wcpredictor.simulation.tournament import STAGES, TeamStats

# openfootball round labels -> the STAGES key naming the round a team *plays in*.
# "Match for third place" is deliberately absent: it does not affect a team's
# deepest stage (both entrants already lost their semifinal) nor the champion.
_ROUND_TO_STAGE = {
    "Round of 32": "round_of_32",
    "Round of 16": "round_of_16",
    "Quarterfinal": "quarterfinal",
    "Quarter-final": "quarterfinal",
    "Quarter-finals": "quarterfinal",
    "Semifinal": "semifinal",
    "Semi-final": "semifinal",
    "Semi-finals": "semifinal",
    "Final": "final",
}
_STAGE_INDEX = {stage: i for i, stage in enumerate(STAGES)}
# Sentinel for "eliminated in / never left the group stage" -- one below R32.
GROUP_STAGE = "group_stage"


class TournamentStateError(ValueError):
    """Raised when the fixtures cannot be parsed into a conditionable state."""


@dataclass(frozen=True)
class KnockoutMatch:
    """A single knockout tie in the frontier round, in bracket order."""

    team1: str
    team2: str
    winner: Optional[str] = None  # set only if the tie has already been played


@dataclass(frozen=True)
class TournamentState:
    """The settled state of the 2026 World Cup, ready to condition a sim on.

    Attributes
    ----------
    reached:
        For every team in the draw, the deepest stage it has already reached
        (a key of :data:`STAGES`, or :data:`GROUP_STAGE` if it never made the
        knockout). Eliminated teams keep this even though they cannot progress.
    group_position:
        Final group finish (1-4) per team, from the completed group results.
    frontier_stage:
        The stage the ``frontier`` teams have reached (e.g. ``"quarterfinal"``).
        Their ties decide who advances to ``frontier_stage``'s successor.
    frontier:
        The current round's ties in standard bracket order -- consecutive
        winners meet in the next round. Flattening ``(team1, team2)`` pairs
        yields the seeding the simulator plays forward from.
    """

    reached: Dict[str, str]
    group_position: Dict[str, int]
    frontier_stage: str
    frontier: Tuple[KnockoutMatch, ...]

    @property
    def alive(self) -> Tuple[str, ...]:
        """Teams still able to win the tournament, in bracket order."""
        return tuple(
            t for m in self.frontier for t in (m.team1, m.team2)
        )

    @property
    def remaining_stages(self) -> Tuple[str, ...]:
        """Stages the frontier winners go on to reach, ending at ``"winner"``."""
        start = _STAGE_INDEX[self.frontier_stage] + 1
        return STAGES[start:]

    def reached_index(self, team: str) -> int:
        """Deepest reached stage as an index into :data:`STAGES` (-1 = group)."""
        return _STAGE_INDEX.get(self.reached.get(team, GROUP_STAGE), -1)


def _is_placeholder(team: Optional[str]) -> bool:
    """openfootball fills undecided knockout slots with tokens like ``"W97"``
    or ``"L101"``; a real country name never contains a digit."""
    return team is None or any(ch.isdigit() for ch in team)


def _match_num(match: dict) -> Optional[int]:
    num = match.get("num")
    try:
        return int(num)
    except (TypeError, ValueError):
        return None


def _ft(match: dict) -> Optional[Tuple[int, int]]:
    ft = (match.get("score") or {}).get("ft")
    if isinstance(ft, list) and len(ft) == 2:
        try:
            return int(ft[0]), int(ft[1])
        except (TypeError, ValueError):
            return None
    return None


def _group_letter(match: dict) -> Optional[str]:
    label = match.get("group")
    if not label:
        return None
    return label.replace("Group", "").strip() or None


def _compute_group_positions(matches: List[dict]) -> Dict[str, int]:
    """Rank each group 1st-4th from its completed round-robin results.

    Ordered by points, then goal difference, then goals for -- the top FIFA
    tiebreakers. Remaining ties fall back to goals against and name so the
    result is deterministic; this is a display aid (the set of qualifiers is
    read directly from the knockout draw, not inferred from this ranking).
    """
    stats: Dict[str, Dict[str, TeamStats]] = {}
    for m in matches:
        letter = _group_letter(m)
        if letter is None or "Matchday" not in (m.get("round") or ""):
            continue
        t1, t2 = m.get("team1"), m.get("team2")
        ft = _ft(m)
        if ft is None or _is_placeholder(t1) or _is_placeholder(t2):
            continue
        hg, ag = ft
        table = stats.setdefault(letter, {})
        sa = table.setdefault(t1, TeamStats())
        sb = table.setdefault(t2, TeamStats())
        sa.gf += hg
        sa.ga += ag
        sb.gf += ag
        sb.ga += hg
        if hg > ag:
            sa.win += 1
            sb.loss += 1
        elif ag > hg:
            sb.win += 1
            sa.loss += 1
        else:
            sa.draw += 1
            sb.draw += 1

    positions: Dict[str, int] = {}
    for table in stats.values():
        ranked = sorted(
            table.items(),
            key=lambda kv: (kv[1].points, kv[1].gd, kv[1].gf, -kv[1].ga, kv[0]),
            reverse=True,
        )
        for pos, (team, _s) in enumerate(ranked, start=1):
            positions[team] = pos
    return positions


def _knockout_by_stage(matches: List[dict]) -> Dict[str, List[dict]]:
    """Group knockout matches by STAGES key, each list sorted by match number."""
    by_stage: Dict[str, List[dict]] = {}
    for m in matches:
        stage = _ROUND_TO_STAGE.get((m.get("round") or "").strip())
        if stage is None:
            continue
        by_stage.setdefault(stage, []).append(m)
    for stage, group in by_stage.items():
        group.sort(key=lambda m: (_match_num(m) is None, _match_num(m) or 0))
    return by_stage


def _winner_of(match: dict, advanced: Dict[int, str]) -> Optional[str]:
    """Winner of a *played* tie, or ``None`` if it has not been played.

    Uses the full-time score when decisive. A level score means the tie was
    settled in extra time or on penalties; the survivor is then whichever side
    shows up (as a real team) in a later round -- captured in ``advanced``,
    which maps a match number to the team that progressed from it.
    """
    t1, t2 = match.get("team1"), match.get("team2")
    if _is_placeholder(t1) or _is_placeholder(t2):
        return None
    ft = _ft(match)
    if ft is None:
        return None
    hg, ag = ft
    if hg > ag:
        return t1
    if ag > hg:
        return t2
    return advanced.get(_match_num(match))


def load_tournament_state(config: Optional[Config] = None) -> TournamentState:
    """Read the vendored 2026 fixtures and distil the settled tournament state.

    Raises :class:`TournamentStateError` if the group stage is only partially
    played (unsupported), if no knockout round has begun, or if the tournament
    is already decided (nothing left to simulate).
    """
    config = config or default_config()
    path: Path = config.wc2026_source_path
    if not path.exists():
        raise FileNotFoundError(
            f"2026 fixtures not found at {path}. Run "
            "scripts/fetch_worldcup_data.py to vendor them."
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    matches: List[dict] = list(doc.get("matches", []))
    for rnd in doc.get("rounds", []):
        matches.extend(rnd.get("matches", []))

    # -- group stage: must be complete before we condition on the knockout.
    group_matches = [m for m in matches if _group_letter(m) is not None
                     and "Matchday" in (m.get("round") or "")]
    unplayed_group = [m for m in group_matches if _ft(m) is None]
    if group_matches and unplayed_group:
        raise TournamentStateError(
            f"{len(unplayed_group)} group match(es) are still unplayed; "
            "conditioning a partially played group stage is not supported."
        )
    group_position = _compute_group_positions(group_matches)

    by_stage = _knockout_by_stage(matches)
    if not by_stage:
        raise TournamentStateError(
            "No knockout matches found; the tournament has not reached the "
            "knockout phase. Use TournamentSimulator.run() to project from "
            "scratch instead."
        )

    # -- how far each team has already gone (deepest round it appears in).
    reached: Dict[str, str] = {}
    for stage in STAGES:
        for m in by_stage.get(stage, []):
            for team in (m.get("team1"), m.get("team2")):
                if _is_placeholder(team):
                    continue
                if _STAGE_INDEX[stage] > _STAGE_INDEX.get(
                    reached.get(team, GROUP_STAGE), -1
                ):
                    reached[team] = stage

    # -- who advanced out of each played tie (so penalty ties resolve). A team
    # advanced from its match iff it turns up, as a real team, one round later.
    advanced: Dict[int, str] = {}
    ordered_stages = [s for s in STAGES if s in by_stage]
    for earlier, later in zip(ordered_stages, ordered_stages[1:]):
        later_teams = {
            t for m in by_stage[later]
            for t in (m.get("team1"), m.get("team2")) if not _is_placeholder(t)
        }
        for m in by_stage[earlier]:
            for t in (m.get("team1"), m.get("team2")):
                if t in later_teams:
                    advanced[_match_num(m)] = t

    # -- the frontier: earliest knockout round with real teams and a match left
    # to play. Placeholder slots (e.g. "W97") are resolved to the winner of the
    # referenced tie so a later round becomes the frontier once its feeders are
    # decided.
    winners: Dict[int, str] = {}
    for stage in ordered_stages:
        for m in by_stage[stage]:
            w = _winner_of(m, advanced)
            if w is not None:
                winners[_match_num(m)] = w

    def resolve(team: Optional[str]) -> Optional[str]:
        if team is None:
            return None
        if not _is_placeholder(team):
            return team
        # "W97" -> winner of match 97 (recursively, in case of chained refs).
        if team[:1] in ("W", "w") and team[1:].isdigit():
            return resolve(winners.get(int(team[1:])))
        return None  # a loser-ref ("L101") or unknown token: not on the path

    frontier_stage: Optional[str] = None
    frontier: List[KnockoutMatch] = []
    for stage in STAGES:
        ties = by_stage.get(stage, [])
        resolved = [
            (resolve(m.get("team1")), resolve(m.get("team2")), m) for m in ties
        ]
        ready = [r for r in resolved if r[0] is not None and r[1] is not None]
        pending = [r for r in ready if _ft(r[2]) is None]
        if ready and pending:
            frontier_stage = stage
            frontier = [
                KnockoutMatch(t1, t2, winners.get(_match_num(m)))
                for t1, t2, m in ready
            ]
            break

    if frontier_stage is None:
        raise TournamentStateError(
            "No knockout round has matches left to play; the tournament is "
            "already decided in the vendored fixtures."
        )
    return TournamentState(
        reached=reached,
        group_position=group_position,
        frontier_stage=frontier_stage,
        frontier=tuple(frontier),
    )
