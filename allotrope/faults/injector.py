"""FaultInjector: apply scripted equipment failures to a running plant.

Wraps a `PolarMicrogridEnv` and enforces each `FaultEvent` at the physical
layer the failure actually belongs to (see `core.py`): a genset that has
tripped is forced offline in the plant's own state and cannot be restarted
by any controller action for the fault's duration; a locked-out battery has
its `max_charge_kw`/`max_discharge_kw` bound to zero at the instance
itself, so no dispatch -- not the agent's, not the safety projection's --
can move power through it; a renewable derate scales the plant's own
precomputed PV/wind availability arrays over the fault window, so
curtailment, reserve margin and everything downstream sees a genuinely
smaller resource, not a controller being lied to about a normal one.

This is deliberately *not* a test of corrupted sensing (stale/NaN/
out-of-range telemetry reaching the controller). That is a separate,
later hardening effort with its own failure model and its own tests;
conflating "the plant is broken" with "the controller is being lied to"
here would test both badly.

Nothing here is wired into training or the safety projection by default:
`FaultInjector` is an opt-in wrapper a benchmark, demo, or test reaches for
explicitly, exactly like `allotrope.optimization`'s MILP oracle is opt-in.
`allotrope.resilience.benchmark` applies these same faults (via `core.py`)
to rule-based and learned controllers evaluated outside Gymnasium
entirely, so a fault means the same thing in both harnesses.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.faults.core import (
    FaultState,
    apply_command_faults,
    apply_renewable_derates,
    unlock_battery,
    validate_targets,
)
from allotrope.faults.events import FaultEvent


class FaultInjector(gym.Wrapper):
    """Replays a fixed schedule of equipment failures against one episode.

    `schedule` step numbers are relative to the episode (0 at `reset()`),
    so the same schedule means the same thing every time it is replayed,
    including against a fresh `PolarMicrogridEnv` with a different
    `randomise_start`.
    """

    def __init__(self, env: PolarMicrogridEnv, schedule: list[FaultEvent]):
        super().__init__(env)
        validate_targets(schedule, env.plant)

        self.schedule = list(schedule)
        self._step_in_episode = 0
        self._fault_state = FaultState()
        self._last_active_faults: list[FaultEvent] = []
        self._original_decode_action = env.decode_action
        env.decode_action = self._decode_action_with_faults  # type: ignore[method-assign]

    # -- gym api ------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._step_in_episode = 0
        for battery in self.env.plant.batteries:
            if battery.id in self._fault_state.locked_battery_ids:
                unlock_battery(battery)
        self._fault_state = FaultState()
        self._last_active_faults = []
        self._pristine_pv = np.array(self.env.plant.pv_available_kw, copy=True)
        self._pristine_wind = np.array(self.env.plant.wind_available_kw, copy=True)
        apply_renewable_derates(
            self.env.plant,
            self.schedule,
            self.env._episode_start,
            self._pristine_pv,
            self._pristine_wind,
        )
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
        command, active = apply_command_faults(
            self.env.plant, self.schedule, self._step_in_episode, command, self._fault_state
        )
        self._last_active_faults = active
        return command


__all__ = ["FaultInjector"]
