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

import datetime
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


def _as_pair(value) -> Optional[Tuple[int, int]]:
    """Coerce an openfootball ``[home, away]`` score list to an int pair."""
    if isinstance(value, list) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _ft(match: dict) -> Optional[Tuple[int, int]]:
    return _as_pair((match.get("score") or {}).get("ft"))


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


def _advanced_map(by_stage: Dict[str, List[dict]]) -> Dict[int, str]:
    """Map a played tie's match number to the team that advanced from it.

    A team advanced from its match iff it turns up, as a real team, one round
    later; this is what lets penalty-shootout ties (level on the full-time
    score) be resolved without a recorded shootout result.
    """
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
    return advanced


def _mask_from_stage(matches: List[dict], as_of_stage: str) -> None:
    """Rewind the fixtures in place to *before* ``as_of_stage`` was played.

    Blanks the scores of the ``as_of_stage`` ties, turning that round into the
    unplayed frontier, and blanks both the scores and the team slots of every
    later knockout round (whose matchups are not yet determined at that point
    in the tournament). Group matches and earlier knockout rounds are left
    untouched. This lets a caller reconstruct the settled state "as of" any
    past round even though the vendored file now holds later results -- the
    basis for the per-phase prediction notebooks.
    """
    target = _STAGE_INDEX[as_of_stage]
    for match in matches:
        stage = _ROUND_TO_STAGE.get((match.get("round") or "").strip())
        if stage is None:
            continue  # group match, or the third-place match (never conditioned on)
        sidx = _STAGE_INDEX[stage]
        if sidx < target:
            continue
        match["score"] = {}  # not yet played
        if sidx > target:
            match["team1"] = None  # matchup not yet determined
            match["team2"] = None


def _winner_of(match: dict, advanced: Dict[int, str]) -> Optional[str]:
    """Winner of a *played* tie, or ``None`` if it has not been played.

    Reads the *settled* score -- after extra time if it was played, otherwise
    the full-time score -- and, when that is still level, the penalty shootout
    (``score.p``). Only if none of those separate the sides does it fall back
    to ``advanced``, which maps a match number to whichever team turned up (as
    a real team) in a later round. That fallback is what let older files omit
    extra-time/shootout detail, but it cannot resolve the *final* -- no round
    follows it -- so a final decided beyond 90 minutes relies on ``score.et``
    or ``score.p`` being recorded (as openfootball does).
    """
    t1, t2 = match.get("team1"), match.get("team2")
    if _is_placeholder(t1) or _is_placeholder(t2):
        return None
    score = match.get("score") or {}
    settled = _as_pair(score.get("et")) or _ft(match)
    if settled is None:
        return None  # not played yet
    hg, ag = settled
    if hg > ag:
        return t1
    if ag > hg:
        return t2
    pens = _as_pair(score.get("p"))
    if pens is not None and pens[0] != pens[1]:
        return t1 if pens[0] > pens[1] else t2
    return advanced.get(_match_num(match))


def _collect_matches(config: Optional[Config] = None) -> Tuple[Config, List[dict]]:
    """Load the vendored 2026 fixtures as a flat list of match dicts.

    Returns the resolved config alongside the matches. Each call re-reads the
    file, so callers are free to mutate the returned dicts (e.g. to rewind the
    tournament to an earlier phase) without affecting anyone else.
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
    return config, matches


def load_tournament_state(
    config: Optional[Config] = None, as_of_stage: Optional[str] = None
) -> TournamentState:
    """Read the vendored 2026 fixtures and distil the settled tournament state.

    By default the frontier is wherever the tournament actually is (the earliest
    knockout round with a match still to play). Pass ``as_of_stage`` (a knockout
    key of :data:`STAGES`, e.g. ``"round_of_32"``) to *rewind* to the start of
    that round: its results -- and every later round's -- are treated as not yet
    played, so the returned state is the "as of the start of that round" view
    used by the per-phase prediction notebooks. Rounds before it stay settled.

    Raises :class:`TournamentStateError` if the group stage is only partially
    played (unsupported), if no knockout round has begun, or if the tournament
    is already decided (nothing left to simulate); raises :class:`ValueError`
    for an ``as_of_stage`` that is not a knockout stage.
    """
    config, matches = _collect_matches(config)
    if as_of_stage is not None:
        if as_of_stage not in _STAGE_INDEX or as_of_stage == "winner":
            raise ValueError(
                "as_of_stage must be a knockout stage in "
                f"{STAGES[:-1]}, got {as_of_stage!r}."
            )
        _mask_from_stage(matches, as_of_stage)

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

    # -- who advanced out of each played tie (so penalty ties resolve).
    advanced = _advanced_map(by_stage)
    ordered_stages = [s for s in STAGES if s in by_stage]

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


def phase_start_dates(
    config: Optional[Config] = None,
) -> Dict[str, datetime.date]:
    """First calendar date of each tournament phase, from the vendored fixtures.

    Keys are :data:`GROUP_STAGE` plus the knockout keys of :data:`STAGES`
    (``"round_of_32"`` ... ``"final"``); each value is the earliest match date
    in that phase. Intended for *point-in-time* model training: to project a
    phase honestly, train only on matches played strictly before its start date
    so the model never "sees" results it is about to predict.
    """
    _config, matches = _collect_matches(config)
    firsts: Dict[str, datetime.date] = {}
    for match in matches:
        raw = match.get("date")
        if not raw:
            continue
        round_label = (match.get("round") or "").strip()
        if "Matchday" in round_label and _group_letter(match) is not None:
            phase = GROUP_STAGE
        else:
            phase = _ROUND_TO_STAGE.get(round_label)
        if phase is None:
            continue
        try:
            day = datetime.date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if phase not in firsts or day < firsts[phase]:
            firsts[phase] = day
    return firsts


def actual_knockout_ties(
    config: Optional[Config] = None,
) -> Dict[str, Tuple[KnockoutMatch, ...]]:
    """The real knockout ties per stage, with the side that actually advanced.

    For every knockout stage present in the vendored fixtures, returns its ties
    (in bracket order) between two real teams, each carrying the ``winner`` that
    progressed -- or ``None`` if the tie has not been played yet. Ties with an
    undetermined slot (a ``"W97"``-style placeholder) are skipped. Used to score
    a phase's predictions against what really happened.
    """
    _config, matches = _collect_matches(config)
    by_stage = _knockout_by_stage(matches)
    advanced = _advanced_map(by_stage)
    out: Dict[str, Tuple[KnockoutMatch, ...]] = {}
    for stage in STAGES:
        ties: List[KnockoutMatch] = []
        for match in by_stage.get(stage, []):
            t1, t2 = match.get("team1"), match.get("team2")
            if _is_placeholder(t1) or _is_placeholder(t2):
                continue
            ties.append(KnockoutMatch(t1, t2, _winner_of(match, advanced)))
        if ties:
            out[stage] = tuple(ties)
    return out
