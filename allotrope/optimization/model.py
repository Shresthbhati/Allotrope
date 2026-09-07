"""The MILP's problem instance: exogenous data plus the physical simplifications made to stay linear.

Every coefficient below is read from the same `StationConfig` and the same
`RewardWeights` prices the RL reward already uses -- nothing here is a new,
invented number. Three real simplifications are made relative to
`allotrope.sim.plant.PolarMicrogrid`, each because the alternative is
nonlinear and this is a *linear* program:

1. **Fuel is the clean Willans line, with no fouling penalty.**
   `Genset.fuel_rate_l_per_h` scales fuel by `1 + 0.05*deposit`, where
   `deposit` is itself a path-dependent state that accumulates from past
   low-load operation. Coupling deposit into the MILP would make fuel a
   product of two decision-dependent quantities across time steps -- no
   longer linear. This MILP uses the clean-engine Willans line
   (`willans_intercept_l_per_h + willans_slope_l_per_kwh * output_kw`) and
   is honest that its fuel numbers are therefore a **lower bound**, not a
   like-for-like comparison to a fouled real engine.
2. **Black carbon uses the clean emission factor only.** The real model's
   wet-stacking penalty (`bc_ef_fouled_mg_per_kwh`, blended in by
   `deposit` and low-load `shortfall`) has the same path-dependency
   problem as (1), for the same reason.
3. **No deposit/wet-stacking accounting at all.** A genset's low-load
   history literally cannot appear in this formulation without carrying
   deposit as an explicit, nonlinearly-coupled state variable across the
   horizon. `wet_stacking_fraction` is therefore not a MILP output -- it
   is not silently zeroed or faked, it is simply not a metric this
   formulation can produce, and `result.py` says so.

Everything else -- generator commitment/output bounds, minimum up/down
time, startup cost, battery SOC dynamics and bounds, reserve margin,
critical-load and firm-thermal service, boiler heat, renewable
availability and curtailment -- is exactly what the simulator itself
enforces or computes, linearised where the simulator's own arithmetic is
already linear (it mostly already is; genset commitment/battery dispatch
were the two genuinely nonlinear couplings, both from (1)/(2) above).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from allotrope.config import StationConfig
from allotrope.sim.plant import PolarMicrogrid

# A large-but-finite sentinel, not float("inf"): CBC requires finite variable
# bounds, and "already satisfied at the horizon boundary" only needs a bound
# that dominates any realistic min-up/min-down window (hours), which this
# dominates by orders of magnitude regardless of `dt_h`. Shared with
# `milp.py`'s own variable bounds so the two never drift apart.
LARGE_STEP_COUNT = 10_000


@dataclass
class MILPInstance:
    """Exogenous data for one MILP horizon. Everything here is known in advance --
    this is what makes the resulting schedule an offline oracle, not something an
    online controller could reproduce without a perfect forecast."""

    cfg: StationConfig
    dt_h: float
    electrical_load_kw: np.ndarray
    critical_load_kw: np.ndarray
    firm_thermal_kw: np.ndarray
    pv_available_kw: np.ndarray
    wind_available_kw: np.ndarray
    initial_soc: dict[str, float] = field(default_factory=dict)
    initial_genset_online: dict[str, bool] = field(default_factory=dict)
    # Steps each genset has already been continuously on/off before t=0,
    # so a min-up/min-down constraint that spans the horizon boundary is
    # enforced correctly rather than assuming every unit just switched.
    initial_online_steps: dict[str, int] = field(default_factory=dict)
    initial_offline_steps: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {
            "electrical_load_kw": len(self.electrical_load_kw),
            "critical_load_kw": len(self.critical_load_kw),
            "firm_thermal_kw": len(self.firm_thermal_kw),
            "pv_available_kw": len(self.pv_available_kw),
            "wind_available_kw": len(self.wind_available_kw),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all exogenous series must share one horizon length: {lengths}")
        if self.horizon == 0:
            raise ValueError("MILPInstance needs at least one time step")

    @property
    def horizon(self) -> int:
        return len(self.electrical_load_kw)


def instance_from_plant(
    plant: PolarMicrogrid, start_index: int, horizon: int
) -> MILPInstance:
    """Build an instance from a live plant's own precomputed weather/demand,
    so the MILP is scored against exactly the same scenario a simulated
    controller would see -- not a separately-generated one that could
    quietly differ."""
    end = start_index + horizon
    if end > plant.n_steps:
        raise ValueError(
            f"horizon [{start_index}, {end}) runs past the plant's {plant.n_steps} steps"
        )

    initial_soc = {b.id: b.state.soc for b in plant.batteries}
    initial_online = {g.id: g.state.online for g in plant.gensets}
    initial_online_steps = {}
    initial_offline_steps = {}
    for g in plant.gensets:
        # A large number, not the exact run length: this only needs to be
        # "at least as long as the longest min-up/min-down window" so the
        # boundary constraint never binds incorrectly for a unit that has
        # simply been in its current state for a long time already.
        if g.state.online:
            initial_online_steps[g.id] = LARGE_STEP_COUNT
            initial_offline_steps[g.id] = 0
        else:
            initial_online_steps[g.id] = 0
            initial_offline_steps[g.id] = LARGE_STEP_COUNT

    return MILPInstance(
        cfg=plant.cfg,
        dt_h=plant.dt_h,
        electrical_load_kw=np.asarray(plant.loads.electrical_kw[start_index:end], dtype=float),
        critical_load_kw=np.asarray(plant.loads.critical_kw[start_index:end], dtype=float),
        firm_thermal_kw=np.asarray(plant.loads.firm_thermal_kw[start_index:end], dtype=float),
        pv_available_kw=np.asarray(plant.pv_available_kw[start_index:end], dtype=float),
        wind_available_kw=np.asarray(plant.wind_available_kw[start_index:end], dtype=float),
        initial_soc=initial_soc,
        initial_genset_online=initial_online,
        initial_online_steps=initial_online_steps,
        initial_offline_steps=initial_offline_steps,
    )


__all__ = ["MILPInstance", "instance_from_plant"]
