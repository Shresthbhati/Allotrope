"""Fault events: the vocabulary this package's injector understands.

Each event names one physical failure mode the plant already has a
representation for in `allotrope.sim.assets`/`allotrope.sim.plant` -- a
tripped genset, a battery too protected to accept or deliver power, a
renewable resource degraded below what the weather alone would produce --
rather than inventing a failure mode the simulator has no model of. Steps
are relative to the *episode*, not the plant's absolute step index, so one
`FaultEvent` means the same thing regardless of where an episode happens to
start (including under `randomise_start`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FaultKind = Literal["genset_trip", "battery_lockout", "renewable_derate"]


@dataclass(frozen=True)
class FaultEvent:
    kind: FaultKind
    start_step: int
    duration_steps: int
    target_id: str | None = None
    """Genset or battery id for `genset_trip`/`battery_lockout`. Must be
    `None` for `renewable_derate`, which is station-wide: nothing in this
    project attributes combined PV+wind availability to one machine."""
    magnitude: float = 1.0
    """Fraction of renewable resource lost, for `renewable_derate` only
    (1.0 = a total whiteout/calm, 0.3 = a 30% derate). Ignored for the
    other two kinds, which are hard, all-or-nothing failures -- a genset
    does not "half trip", and this project already has a full continuous
    model of partial generator output (ordinary load-following dispatch),
    so a fractional trip would just duplicate that under a different
    name."""

    def __post_init__(self) -> None:
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        if self.kind in ("genset_trip", "battery_lockout") and self.target_id is None:
            raise ValueError(f"{self.kind} needs a target_id")
        if self.kind == "renewable_derate":
            if self.target_id is not None:
                raise ValueError("renewable_derate is station-wide; target_id must be None")
            if not (0.0 < self.magnitude <= 1.0):
                raise ValueError("renewable_derate magnitude must be in (0, 1]")

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration_steps

    def active_at(self, step: int) -> bool:
        return self.start_step <= step < self.end_step


__all__ = ["FaultEvent", "FaultKind"]
