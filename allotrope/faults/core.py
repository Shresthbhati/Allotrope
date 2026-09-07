"""Shared fault-application logic.

Used identically by `FaultInjector` (the Gymnasium wrapper, for RL
training/evaluation) and `allotrope.resilience.benchmark` (which drives
rule-based and learned controllers through `allotrope.sim.runner.run_episode`
directly, bypassing Gymnasium entirely). One implementation, so a fault
means the same physical thing regardless of which harness applied it --
see `injector.py`'s module docstring for why each fault is enforced where
it is.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field, replace

import numpy as np

from allotrope.faults.events import FaultEvent
from allotrope.sim.assets import Battery
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid


def _locked_max_kw(self: Battery, dt_h: float = 0.25) -> float:  # noqa: ARG001
    return 0.0


def lock_battery(battery: Battery) -> None:
    battery.max_charge_kw = types.MethodType(_locked_max_kw, battery)
    battery.max_discharge_kw = types.MethodType(_locked_max_kw, battery)


def unlock_battery(battery: Battery) -> None:
    # Deleting the instance-level override restores normal lookup of
    # Battery.max_charge_kw/max_discharge_kw from the class.
    del battery.max_charge_kw
    del battery.max_discharge_kw


@dataclass
class FaultState:
    """The one piece of state fault application needs across steps of an
    episode: which batteries are currently locked, so a lock is applied
    exactly once (on the transition into the fault) and released exactly
    once (on the transition out)."""

    locked_battery_ids: set[str] = field(default_factory=set)


def apply_command_faults(
    plant: PolarMicrogrid,
    schedule: list[FaultEvent],
    step: int,
    command: DispatchCommand,
    state: FaultState,
) -> tuple[DispatchCommand, list[FaultEvent]]:
    """Enforce every fault active at `step` (episode-relative) against
    `command` and the plant's own genset/battery state, and return the
    (possibly rewritten) command plus the events active at this step."""
    active = [e for e in schedule if e.active_at(step)]

    tripped_ids = {e.target_id for e in active if e.kind == "genset_trip"}
    if tripped_ids:
        genset_on = list(command.genset_on)
        setpoints = list(command.genset_setpoint_kw)
        for k, g in enumerate(plant.gensets):
            if g.id not in tripped_ids:
                continue
            genset_on[k] = False
            setpoints[k] = 0.0
            just_started_tripping = not any(
                e.kind == "genset_trip" and e.target_id == g.id and e.active_at(step - 1)
                for e in schedule
            )
            if just_started_tripping:
                # A trip is a hardware failure, not a commanded stop: it
                # does not wait on the anti-cycling minimum-up-time a
                # normal shutdown request would.
                g.state.online = False
                g.state.power_kw = 0.0
        command = replace(
            command, genset_on=tuple(genset_on), genset_setpoint_kw=tuple(setpoints)
        )

    locked_ids = {e.target_id for e in active if e.kind == "battery_lockout"}
    newly_locked = locked_ids - state.locked_battery_ids
    newly_unlocked = state.locked_battery_ids - locked_ids
    for b in plant.batteries:
        if b.id in newly_locked:
            lock_battery(b)
        elif b.id in newly_unlocked:
            unlock_battery(b)
    state.locked_battery_ids = locked_ids

    if locked_ids:
        battery_kw = list(command.battery_kw)
        for k, b in enumerate(plant.batteries):
            if b.id in locked_ids:
                battery_kw[k] = 0.0
        command = replace(command, battery_kw=tuple(battery_kw))

    return command, active


def apply_renewable_derates(
    plant: PolarMicrogrid,
    schedule: list[FaultEvent],
    episode_start: int,
    pristine_pv: np.ndarray,
    pristine_wind: np.ndarray,
) -> None:
    """Overwrite `plant.pv_available_kw`/`wind_available_kw` in place from
    the given pristine arrays, applying every `renewable_derate` event in
    `schedule`. Call once, right after the arrays are known pristine and
    before stepping -- `plant.reset()` does not touch these arrays, so
    calling this immediately after building the plant (before any
    `reset()`) is safe."""
    plant.pv_available_kw = pristine_pv.copy()
    plant.wind_available_kw = pristine_wind.copy()
    for event in schedule:
        if event.kind != "renewable_derate":
            continue
        lo = max(episode_start + event.start_step, 0)
        hi = min(episode_start + event.end_step, plant.n_steps)
        if lo < hi:
            plant.pv_available_kw[lo:hi] *= 1.0 - event.magnitude
            plant.wind_available_kw[lo:hi] *= 1.0 - event.magnitude


def validate_targets(schedule: list[FaultEvent], plant: PolarMicrogrid) -> None:
    genset_ids = {g.id for g in plant.gensets}
    storage_ids = {b.id for b in plant.batteries}
    for event in schedule:
        if event.kind == "genset_trip" and event.target_id not in genset_ids:
            raise ValueError(f"unknown genset id {event.target_id!r}")
        if event.kind == "battery_lockout" and event.target_id not in storage_ids:
            raise ValueError(f"unknown storage id {event.target_id!r}")


__all__ = [
    "FaultState",
    "apply_command_faults",
    "apply_renewable_derates",
    "validate_targets",
    "lock_battery",
    "unlock_battery",
]
