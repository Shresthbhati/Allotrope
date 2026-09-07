"""Run controllers against fault scenarios and report how each holds up.

Reuses `allotrope.sim.runner.run_episode` -- the same entry point every
controller in this project is already evaluated through -- via its
`command_hook`, so a benchmarked run differs from an ordinary evaluation
run in exactly one respect: the scripted faults. Everything else (how the
plant works, how a controller is invoked, how the summary is computed) is
identical, which is what makes a comparison between controllers, or
between a scenario and its `baseline`, mean anything.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from allotrope.config import StationConfig
from allotrope.faults.core import FaultState, apply_command_faults, apply_renewable_derates
from allotrope.faults.events import FaultEvent
from allotrope.resilience.scenarios import STANDARD_SCENARIOS
from allotrope.sim.plant import PolarMicrogrid
from allotrope.sim.runner import EpisodeResult, build_plant, run_episode


def run_scenario(
    controller: Any,
    cfg: StationConfig,
    schedule: list[FaultEvent],
    *,
    start: str = "2026-06-01",
    periods: int = 336,
    seed: int = 0,
) -> EpisodeResult:
    """Run one controller through one fault schedule and return the full result.

    `periods` defaults to a week (336 hourly steps) -- long enough to hold
    every `resilience.scenarios` schedule's fault window with days of
    recovery on either side, short enough to benchmark many
    (controller, scenario) pairs without each one being a multi-hour run.
    """
    plant: PolarMicrogrid = build_plant(cfg, start=start, periods=periods, seed=seed)
    pristine_pv = np.array(plant.pv_available_kw, copy=True)
    pristine_wind = np.array(plant.wind_available_kw, copy=True)
    apply_renewable_derates(plant, schedule, episode_start=0, pristine_pv=pristine_pv, pristine_wind=pristine_wind)

    fault_state = FaultState()

    def hook(step, command):
        command, _active = apply_command_faults(plant, schedule, step, command, fault_state)
        return command

    return run_episode(plant, controller, max_steps=periods, command_hook=hook)


def run_benchmark(
    controllers: dict[str, Callable[[StationConfig], Any]],
    cfg: StationConfig,
    scenarios: dict[str, Callable[[StationConfig], list[FaultEvent]]] | None = None,
    **scenario_kwargs: Any,
) -> pd.DataFrame:
    """One row per (controller, scenario), including a `baseline` row (no
    faults) for every controller, so a fault's cost on a given controller
    is legible as the difference between its `baseline` row and the
    scenario's row -- everything else about the run is held fixed.

    `scenarios` defaults to `resilience.scenarios.STANDARD_SCENARIOS`
    (which already includes `baseline`); pass a smaller dict to benchmark
    a subset, but include `"baseline"` explicitly if you do, since nothing
    here injects it automatically.
    """
    scenarios = scenarios if scenarios is not None else STANDARD_SCENARIOS
    rows: list[dict[str, Any]] = []
    for controller_name, make_controller in controllers.items():
        for scenario_name, make_schedule in scenarios.items():
            schedule = make_schedule(cfg)
            controller = make_controller(cfg)
            result = run_scenario(controller, cfg, schedule, **scenario_kwargs)
            row = {"controller": controller_name, "scenario": scenario_name}
            row.update(result.summary)
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["run_scenario", "run_benchmark"]
