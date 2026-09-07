"""FaultInjector: apply scripted equipment failures to a running plant.

Wraps a `PolarMicrogridEnv` and enforces each `FaultEvent` at the physical
layer the failure actually belongs to: a genset that has tripped is forced
offline in the plant's own state (`GensetState.online`) and cannot be
restarted by any controller action for the fault's duration; a locked-out
battery has its `max_charge_kw`/`max_discharge_kw` bound to zero at the
instance itself, so no dispatch -- not the agent's, not the safety
projection's -- can move power through it; a renewable derate scales the
plant's own precomputed PV/wind availability arrays over the fault window,
so curtailment, reserve margin and everything downstream sees a genuinely
smaller resource, not a controller being lied to about a normal one.

This is deliberately *not* a test of corrupted sensing (stale/NaN/
out-of-range telemetry reaching the controller). That is a separate,
later hardening effort with its own failure model and its own tests;
conflating "the plant is broken" with "the controller is being lied to"
here would test both badly.

Nothing here is wired into training or the safety projection by default:
`FaultInjector` is an opt-in wrapper a benchmark, demo, or test reaches for
explicitly, exactly like `allotrope.optimization`'s MILP oracle is opt-in.
"""

from __future__ import annotations

import types
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.faults.events import FaultEvent
from allotrope.sim.assets import Battery


def _locked_max_kw(self: Battery, dt_h: float = 0.25) -> float:  # noqa: ARG001
    return 0.0


def _lock_battery(battery: Battery) -> None:
    battery.max_charge_kw = types.MethodType(_locked_max_kw, battery)
    battery.max_discharge_kw = types.MethodType(_locked_max_kw, battery)


def _unlock_battery(battery: Battery) -> None:
    # Deleting the instance-level override restores normal lookup of
    # Battery.max_charge_kw/max_discharge_kw from the class.
    del battery.max_charge_kw
    del battery.max_discharge_kw


class FaultInjector(gym.Wrapper):
    """Replays a fixed schedule of equipment failures against one episode.

    `schedule` step numbers are relative to the episode (0 at `reset()`),
    so the same schedule means the same thing every time it is replayed,
    including against a fresh `PolarMicrogridEnv` with a different
    `randomise_start`.
    """

    def __init__(self, env: PolarMicrogridEnv, schedule: list[FaultEvent]):
        super().__init__(env)
        genset_ids = {g.id for g in env.cfg.gensets}
        storage_ids = {s.id for s in env.cfg.storage}
        for event in schedule:
            if event.kind == "genset_trip" and event.target_id not in genset_ids:
                raise ValueError(f"unknown genset id {event.target_id!r}")
            if event.kind == "battery_lockout" and event.target_id not in storage_ids:
                raise ValueError(f"unknown storage id {event.target_id!r}")

        self.schedule = list(schedule)
        self._step_in_episode = 0
        self._locked_battery_ids: set[str] = set()
        self._last_active_faults: list[FaultEvent] = []
        self._original_decode_action = env.decode_action
        env.decode_action = self._decode_action_with_faults  # type: ignore[method-assign]

    # -- gym api ------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._step_in_episode = 0
        for battery in self.env.plant.batteries:
            if battery.id in self._locked_battery_ids:
                _unlock_battery(battery)
        self._locked_battery_ids = set()
        self._last_active_faults = []
        self._pristine_pv = np.array(self.env.plant.pv_available_kw, copy=True)
        self._pristine_wind = np.array(self.env.plant.wind_available_kw, copy=True)
        self._apply_renewable_derates()
        obs = self.env._observe()
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["active_faults"] = [e.kind for e in self._last_active_faults]
        self._step_in_episode += 1
        return obs, reward, terminated, truncated, info

    # -- fault application ---------------------------------------------

    def _decode_action_with_faults(self, action: dict[str, Any], obs: Any = None):
        command = self._original_decode_action(action, obs)
        step = self._step_in_episode
        active = [e for e in self.schedule if e.active_at(step)]
        self._last_active_faults = active
        plant = self.env.plant

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
                    for e in self.schedule
                )
                if just_started_tripping:
                    # A trip is a hardware failure, not a commanded stop: it
                    # does not wait on the anti-cycling minimum-up-time the
                    # way a normal shutdown request would.
                    g.state.online = False
                    g.state.power_kw = 0.0
            command = replace(
                command, genset_on=tuple(genset_on), genset_setpoint_kw=tuple(setpoints)
            )

        locked_ids = {e.target_id for e in active if e.kind == "battery_lockout"}
        newly_locked = locked_ids - self._locked_battery_ids
        newly_unlocked = self._locked_battery_ids - locked_ids
        for b in plant.batteries:
            if b.id in newly_locked:
                _lock_battery(b)
            elif b.id in newly_unlocked:
                _unlock_battery(b)
        self._locked_battery_ids = locked_ids

        if locked_ids:
            battery_kw = list(command.battery_kw)
            for k, b in enumerate(plant.batteries):
                if b.id in locked_ids:
                    battery_kw[k] = 0.0
            command = replace(command, battery_kw=tuple(battery_kw))

        return command

    def _apply_renewable_derates(self) -> None:
        plant = self.env.plant
        plant.pv_available_kw = self._pristine_pv.copy()
        plant.wind_available_kw = self._pristine_wind.copy()
        start_index = self.env._episode_start
        for event in self.schedule:
            if event.kind != "renewable_derate":
                continue
            lo = max(start_index + event.start_step, 0)
            hi = min(start_index + event.end_step, plant.n_steps)
            if lo < hi:
                plant.pv_available_kw[lo:hi] *= 1.0 - event.magnitude
                plant.wind_available_kw[lo:hi] *= 1.0 - event.magnitude


__all__ = ["FaultInjector"]
