"""Named fault schedules for the resilience benchmark.

Each function takes a `StationConfig` and returns a `list[FaultEvent]`, so
the same scenario can be built against any station without hardcoding a
genset or battery id that might not exist there. Steps are episode-relative
(0 at the start of the run), at the project's standard 1-hour dispatch
interval unless a caller passes a different one.
"""

from __future__ import annotations

from typing import Callable

from allotrope.config import StationConfig
from allotrope.faults.events import FaultEvent


def no_faults(cfg: StationConfig) -> list[FaultEvent]:
    """The control: every scenario's cost is measured against this baseline."""
    return []


def single_genset_trip(cfg: StationConfig) -> list[FaultEvent]:
    """The largest set fails for a full day, mid-run."""
    largest = max(cfg.gensets, key=lambda g: g.rated_kw)
    return [FaultEvent(kind="genset_trip", start_step=48, duration_steps=24, target_id=largest.id)]


def battery_lockout_at_peak(cfg: StationConfig) -> list[FaultEvent]:
    """Storage is unavailable for half a day, starting where a rule-based
    controller would normally lean on it hardest (the ordinary evening
    demand peak, here approximated as hour 18 of the run)."""
    battery = cfg.storage[0]
    return [FaultEvent(kind="battery_lockout", start_step=18, duration_steps=12, target_id=battery.id)]


def extended_whiteout(cfg: StationConfig) -> list[FaultEvent]:
    """Three days of severely reduced renewable resource -- a persistent
    storm or a multi-day polar-night stretch's worth of near-zero PV
    compounded with slack wind, not a single-hour dip."""
    return [FaultEvent(kind="renewable_derate", start_step=24, duration_steps=72, magnitude=0.85)]


def compound_failure(cfg: StationConfig) -> list[FaultEvent]:
    """The scenario the other three are meant to bound: a genset trip
    landing in the middle of a multi-day renewable shortfall, so the plant
    loses generating capacity exactly when it can least spare it."""
    largest = max(cfg.gensets, key=lambda g: g.rated_kw)
    return [
        FaultEvent(kind="renewable_derate", start_step=12, duration_steps=60, magnitude=0.8),
        FaultEvent(kind="genset_trip", start_step=36, duration_steps=18, target_id=largest.id),
    ]


STANDARD_SCENARIOS: dict[str, Callable[[StationConfig], list[FaultEvent]]] = {
    "baseline": no_faults,
    "single_genset_trip": single_genset_trip,
    "battery_lockout_at_peak": battery_lockout_at_peak,
    "extended_whiteout": extended_whiteout,
    "compound_failure": compound_failure,
}

__all__ = [
    "no_faults",
    "single_genset_trip",
    "battery_lockout_at_peak",
    "extended_whiteout",
    "compound_failure",
    "STANDARD_SCENARIOS",
]
