"""The MILP's output: one structured, inspectable result, not a bare number.

Every array is `(horizon, n_units)` (or `(horizon,)` for station-level
series), in the same units `allotrope.sim.plant.PolarMicrogrid.step`'s
telemetry uses, so a result can be compared line-for-line against a
simulated controller's run without a unit-conversion step that could hide
a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizationResult:
    status: str
    """PuLP's own solver status string, e.g. "Optimal", "Infeasible", "Not Solved"
    (the last meaning a timeout with no feasible incumbent found yet)."""

    objective_value: float | None
    solve_time_s: float
    dt_h: float

    genset_ids: list[str]
    storage_ids: list[str]

    genset_on: np.ndarray
    """(T, n_g) binary."""
    genset_output_kw: np.ndarray
    """(T, n_g)."""
    genset_started: np.ndarray
    """(T, n_g) binary -- 1 exactly on a genset's off-to-on transition."""

    battery_charge_kw: np.ndarray
    """(T, n_s), >= 0."""
    battery_discharge_kw: np.ndarray
    """(T, n_s), >= 0."""
    battery_soc: np.ndarray
    """(T+1, n_s) -- includes the initial SOC at index 0."""

    boiler_heat_kw: np.ndarray
    """(T,)."""
    renewable_used_kw: np.ndarray
    """(T,)."""
    curtailed_kw: np.ndarray
    """(T,)."""

    fuel_l: np.ndarray
    """(T,) genset + boiler fuel, litres per step."""
    black_carbon_g: np.ndarray
    """(T,), clean-engine emission factor only -- see model.py's module docstring."""

    @property
    def feasible(self) -> bool:
        return self.status == "Optimal"

    def summary(self) -> dict[str, float]:
        """Episode totals, in the same key names as `PolarMicrogrid.summary()`
        wherever this formulation computes the equivalent quantity -- so a
        result can be dropped into the same comparison table a simulated
        controller's `summary()` output already goes into. Keys with no
        MILP equivalent (wet_stacking_fraction, mean_deposit) are simply
        absent rather than filled with an invented zero."""
        total_starts = int(self.genset_started.sum())
        genset_kwh = float((self.genset_output_kw * self.dt_h).sum())
        run_hours = float((self.genset_on * self.dt_h).sum())
        return {
            "fuel_l": float(self.fuel_l.sum()),
            "fuel_kl": float(self.fuel_l.sum()) / 1000.0,
            "black_carbon_g": float(self.black_carbon_g.sum()),
            "genset_starts": total_starts,
            "genset_run_hours": run_hours,
            "genset_kwh": genset_kwh,
            "renewable_kwh": float((self.renewable_used_kw * self.dt_h).sum()),
            "curtailed_kwh": float((self.curtailed_kw * self.dt_h).sum()),
            "solve_time_s": self.solve_time_s,
        }


__all__ = ["OptimizationResult"]
