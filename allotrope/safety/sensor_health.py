"""Per-sensor staleness and plausibility classification.

This is layered *on top of* `projection.py`'s existing NaN/Inf
sanitisation (`SafetyProjection._sanitise_observation`), not a replacement
for it. That sanitiser already catches the failure mode this project's own
adversarial audit found: a non-finite reading reaching a bound uncaught,
and its fail-toward-conservative substitution is already correct and
already tested -- nothing here changes it.

What the existing sanitiser cannot catch is a sensor that is *finite* but
wrong: frozen at its last good value (a stuck ADC, a dead comms link
papering over silence with the last frame it received) or reporting a
value that is finite but physically impossible for that quantity (a
battery SOC of 4.0, a negative critical load). Neither trips a NaN check,
and both are real failure modes on real telemetry links -- a link that has
died reports nothing new, and a controller reading the last good frame
forever cannot tell that from a station whose load has stopped changing.

This module only classifies and names the problem, for an operator/HMI or
a caller that wants to react to a degrading sensor (e.g. by falling back
to `allotrope.safety.fallback.GuardedController`) before the reading is
bad enough to be caught by range or NaN checks. It does not alter the
observation, the command, or the plant -- `SafetyProjection` is still the
only thing with the authority to change what reaches the plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from allotrope.config import StationConfig


class SensorStatus(str, Enum):
    """Ordered worst-to-best is CORRUPTED > STALE > DEGRADED > VALID.
    `UNSAFE` is reserved for a caller that wants to escalate repeated or
    combined CORRUPTED/STALE findings into a station-level alarm; `classify`
    never emits it itself, since it grades one reading at a time and has no
    notion of "how many other sensors are also bad right now."
    """

    VALID = "valid"
    DEGRADED = "degraded"
    STALE = "stale"
    CORRUPTED = "corrupted"
    UNSAFE = "unsafe"


_RANK = {
    SensorStatus.VALID: 0,
    SensorStatus.DEGRADED: 1,
    SensorStatus.STALE: 2,
    SensorStatus.CORRUPTED: 3,
    SensorStatus.UNSAFE: 4,
}


def worse(a: SensorStatus, b: SensorStatus) -> SensorStatus:
    return a if _RANK[a] >= _RANK[b] else b


@dataclass(frozen=True)
class SensorSpec:
    """What "plausible" means for one telemetry field.

    Bounds should come from the station's own physical limits wherever one
    exists (a rated capacity, a chemistry's SOC envelope) -- see
    `specs_for_station` -- not an invented number.
    """

    key: str
    lo: float
    hi: float
    max_stale_steps: int
    """Consecutive bit-identical readings tolerated before a signal that
    should be moving (weather, time-varying demand) is judged stuck rather
    than genuinely flat. `0` disables staleness checking for a field that
    can legitimately hold still (e.g. a fixed rated capacity)."""


@dataclass
class SensorHistory:
    """One field's rolling state: its last value and how long it has held there."""

    last_value: float | None = None
    repeat_count: int = 0

    def update(self, value: float) -> int:
        if self.last_value is not None and value == self.last_value:
            self.repeat_count += 1
        else:
            self.last_value = value
            self.repeat_count = 0
        return self.repeat_count


def classify(value: Any, spec: SensorSpec, history: SensorHistory) -> SensorStatus:
    """Grade one reading against `spec` and advance `history` in place.

    Call once per step, per field, in step order -- staleness is defined
    relative to the previous call, so skipping steps or calling out of
    order gives a meaningless repeat count.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not np.isfinite(v) or v < spec.lo or v > spec.hi:
        # A corrupted reading still advances history (as "inf", never
        # matching a prior finite value), so a sensor that starts failing
        # does not also get credit toward a later staleness count once it
        # (implausibly) starts reporting a constant garbage value.
        history.update(float("inf"))
        return SensorStatus.CORRUPTED
    repeats = history.update(v)
    if spec.max_stale_steps and repeats >= spec.max_stale_steps:
        return SensorStatus.STALE
    return SensorStatus.VALID


@dataclass
class ObservationHealth:
    """Grades every configured field of one station's observation, step by step.

    One instance per episode/run: it owns the rolling per-field history a
    staleness judgement needs, so a fresh instance must be used for a fresh
    episode (reusing one across episodes would misread the last episode's
    final readings as this episode's stale ones).
    """

    specs: dict[str, SensorSpec]
    _histories: dict[str, SensorHistory] = field(default_factory=dict)

    def check(self, observation: dict) -> dict[str, SensorStatus]:
        statuses: dict[str, SensorStatus] = {}
        for key, spec in self.specs.items():
            if key not in observation:
                statuses[key] = SensorStatus.CORRUPTED
                continue
            value = observation[key]
            if isinstance(value, (list, tuple, np.ndarray)):
                worst = SensorStatus.VALID
                for i, v in enumerate(value):
                    sub_history = self._histories.setdefault(f"{key}[{i}]", SensorHistory())
                    worst = worse(worst, classify(v, spec, sub_history))
                statuses[key] = worst
            else:
                history = self._histories.setdefault(key, SensorHistory())
                statuses[key] = classify(value, spec, history)
        return statuses

    def worst(self, observation: dict) -> SensorStatus:
        statuses = self.check(observation)
        result = SensorStatus.VALID
        for status in statuses.values():
            result = worse(result, status)
        return result


def specs_for_station(cfg: StationConfig, max_stale_steps: int = 0) -> dict[str, SensorSpec]:
    """Default plausibility bounds for the fields `SafetyProjection`
    actually reads from `plant.observe()` (see `_sanitise_observation`).

    `max_stale_steps` defaults to `0` (staleness checking off) for every
    field here, on real evidence rather than caution for its own sake: a
    run against this project's own synthetic Maitri telemetry showed
    `critical_load_kw` (a flat baseload for hours at a time),
    `pv_available_kw` (genuinely, correctly zero for the many consecutive
    hours of a polar night), and a battery's `max_charge_kw` (pinned at
    its nameplate cap whenever SOC and temperature aren't the binding
    constraint) all holding an exact value for far longer than any
    threshold that would still catch a stuck sensor promptly. This
    simulator emits clean values with no sensor noise, so "unchanged for
    N steps" cannot separate a genuinely constant physical state from a
    frozen reading here -- staleness detection needs either injected
    sensor noise or a real (noisy) telemetry feed to calibrate a threshold
    against, neither of which this project has yet. The mechanism is kept
    and tested (`tests/test_sensor_health.py`) for a caller with real
    telemetry to enable per-field via `SensorSpec.max_stale_steps`; it is
    simply not turned on here until there is a real signal to tune it on.
    """
    max_genset_kw = sum(g.rated_kw for g in cfg.gensets)
    max_pv_kw = cfg.pv.rated_kwp
    max_wind_kw = cfg.wind.rated_kw_total
    max_recoverable_heat_kw = sum(g.rated_kw * g.chp_heat_ratio for g in cfg.gensets)
    max_thermal_kw = cfg.thermal.boiler_rated_kw + max_recoverable_heat_kw
    # A generous multiple of installed capacity, not a tight bound: this
    # is a garbage-value catch (a corrupted reading orders of magnitude
    # off), not a precise load forecast.
    max_load_kw = 3.0 * max(max_genset_kw, 1.0)

    specs = {
        "critical_load_kw": SensorSpec("critical_load_kw", 0.0, max_load_kw, max_stale_steps),
        "firm_thermal_kw": SensorSpec("firm_thermal_kw", 0.0, max_thermal_kw * 3.0, max_stale_steps),
        "pv_available_kw": SensorSpec("pv_available_kw", 0.0, max_pv_kw * 1.05, max_stale_steps),
        "wind_available_kw": SensorSpec("wind_available_kw", 0.0, max_wind_kw * 1.05, max_stale_steps),
        "battery_max_charge_kw": SensorSpec(
            "battery_max_charge_kw", 0.0, max((s.max_charge_kw for s in cfg.storage), default=0.0) * 1.05, max_stale_steps
        ),
        "battery_max_discharge_kw": SensorSpec(
            "battery_max_discharge_kw",
            0.0,
            max((s.max_discharge_kw for s in cfg.storage), default=0.0) * 1.05,
            max_stale_steps,
        ),
        "battery_soc": SensorSpec("battery_soc", -0.01, 1.01, 0),
    }
    return specs


__all__ = [
    "SensorStatus",
    "SensorSpec",
    "SensorHistory",
    "ObservationHealth",
    "classify",
    "worse",
    "specs_for_station",
]
