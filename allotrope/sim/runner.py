"""Run a controller against a plant and collect the results.

One entry point, used identically by the rule-based baselines, the learned
agents and the evaluation harness, so that no comparison between them can be
contaminated by a difference in how they were run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from allotrope.config import StationConfig, load_station
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid
from allotrope.synth.climate import ClimateGenerator, ClimateSeries
from allotrope.synth.loads import LoadGenerator, LoadSeries


@dataclass
class EpisodeResult:
    """Everything one run produced: the trace, the headline numbers, the plant."""

    telemetry: pd.DataFrame
    summary: dict[str, float]
    controller: str
    plant: PolarMicrogrid

    def daily(self) -> pd.DataFrame:
        """Daily aggregates, the resolution most operational questions are asked at."""
        df = self.telemetry
        return pd.DataFrame(
            {
                "fuel_l": df["fuel_l"].resample("1D").sum(),
                "black_carbon_g": df["black_carbon_mg"].resample("1D").sum() / 1000.0,
                "load_kwh": (df["electrical_load_kw"] * self.plant.dt_h).resample("1D").sum(),
                "renewable_kwh": (df["renewable_used_kw"] * self.plant.dt_h).resample("1D").sum(),
                "curtailed_kwh": (df["curtailed_kw"] * self.plant.dt_h).resample("1D").sum(),
                "wet_stacking_frac": df["wet_stacking"].resample("1D").mean(),
                "mean_deposit": df["mean_deposit"].resample("1D").mean(),
                "min_indoor_temp_c": df["indoor_temp_c"].resample("1D").min(),
            }
        )


def build_plant(
    station: str | StationConfig = "maitri",
    start: str = "2026-01-01",
    periods: int = 8760,
    freq: str = "1h",
    seed: int | None = 0,
) -> PolarMicrogrid:
    """Assemble a plant with a fresh weather and demand realisation.

    The seed is the only thing that distinguishes one realisation from another,
    which is what makes held-out evaluation seeds meaningful later.
    """
    cfg = station if isinstance(station, StationConfig) else load_station(station)
    climate = ClimateGenerator(cfg, seed=seed).generate(start, periods, freq)
    loads = LoadGenerator(cfg, seed=None if seed is None else seed + 10_000).generate(climate)
    return PolarMicrogrid(cfg, climate, loads)


def run_episode(
    plant: PolarMicrogrid,
    controller: Any,
    max_steps: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    command_hook: Callable[[int, DispatchCommand], DispatchCommand] | None = None,
) -> EpisodeResult:
    """Step the plant under a controller until the weather series runs out.

    `command_hook`, when given, is called as `command_hook(step, command)`
    after the controller decides and before the plant executes, and its
    return value is what actually reaches `plant.step`. This is the one
    seam this shared entry point offers for a caller that needs to alter
    what the plant sees without altering the controller under test --
    `allotrope.resilience.benchmark` is the motivating use, applying
    scripted equipment faults identically to every controller compared.
    """
    plant.reset()
    if hasattr(controller, "reset"):
        controller.reset()

    limit = plant.n_steps if max_steps is None else min(max_steps, plant.n_steps)
    records: list[dict[str, Any]] = []

    for step in range(limit):
        observation = plant.observe()
        command = controller.act(observation, plant)
        if command_hook is not None:
            command = command_hook(step, command)
        telemetry = plant.step(command)
        records.append(_flatten(telemetry))
        if progress is not None and step % 500 == 0:
            progress(step, limit)

    frame = pd.DataFrame.from_records(records).set_index("timestamp")
    return EpisodeResult(
        telemetry=frame,
        summary=plant.summary(),
        controller=getattr(controller, "name", type(controller).__name__),
        plant=plant,
    )


def _flatten(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Expand per-unit lists into columns, so the trace is a flat table."""
    flat: dict[str, Any] = {}
    for key, value in telemetry.items():
        if isinstance(value, list):
            for unit, item in enumerate(value):
                flat[f"{key}_{unit}"] = item
            if key == "genset_deposit":
                flat["mean_deposit"] = sum(value) / max(len(value), 1)
            if key == "genset_load_frac":
                online = [v for v in value if v > 0.0]
                flat["mean_online_load_frac"] = sum(online) / len(online) if online else 0.0
        else:
            flat[key] = value
    return flat


def compare(results: list[EpisodeResult], keys: list[str] | None = None) -> pd.DataFrame:
    """Put several runs side by side on the metrics the project is judged on."""
    keys = keys or [
        "fuel_kl",
        "black_carbon_g",
        "specific_fuel_l_per_kwh",
        "mean_genset_load_frac",
        "wet_stacking_fraction",
        "mean_deposit",
        "renewable_fraction",
        "curtailed_kwh",
        "genset_run_hours",
        "genset_starts",
        "critical_unserved_kwh",
        "freeze_violation_steps",
        "unmet_water_kwh",
    ]
    return pd.DataFrame(
        {r.controller: {k: r.summary.get(k, float("nan")) for k in keys} for r in results}
    )


DEFAULT_COMPARISON_KEYS = [
    "fuel_kl",
    "black_carbon_g",
    "mean_genset_load_frac",
    "wet_stacking_fraction",
    "renewable_fraction",
    "genset_starts",
    "critical_unserved_kwh",
    "freeze_violation_steps",
]


def compare_multi(
    results_by_label: dict[str, list[EpisodeResult]], keys: list[str] | None = None
) -> pd.DataFrame:
    """Mean-across-seeds comparison table: one column per label, one row per metric.

    This is the multi-seed counterpart to `compare` -- averaging each label's
    list of episode results (e.g. one per held-out seed) before laying them out
    the same way, metrics as rows and labels as columns. Kept here rather than
    duplicated in a reporting script, because a script's own DataFrame-shape
    logic is exactly the kind of code that looks obviously right and silently
    is not: an earlier, untested version of this in scripts/evaluate_agent.py
    built the same table with rows and columns transposed.
    """
    keys = keys or DEFAULT_COMPARISON_KEYS
    columns = {}
    for label, results in results_by_label.items():
        if not results:
            raise ValueError(f"{label!r} has no results to average")
        columns[label] = {
            k: sum(r.summary.get(k, float("nan")) for r in results) / len(results) for k in keys
        }
    return pd.DataFrame(columns)


__all__ = ["build_plant", "run_episode", "compare", "EpisodeResult"]
