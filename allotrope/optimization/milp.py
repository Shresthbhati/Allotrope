"""The MILP itself: a mixed-integer linear program built with PuLP, solved with CBC.

`solve(instance, timeout_s=None) -> OptimizationResult` is the entire public
interface. Constraint and objective construction are kept in this one module
rather than split into separate `constraints.py`/`objective.py` files (as a
first sketch of this package considered) -- at this formulation's size,
splitting them would mean passing the same half-dozen PuLP variable
dictionaries across three files for no real separation of concerns. If this
formulation grows a second use (e.g. a rolling-horizon MPC reusing pieces of
it), that is the point to factor out a shared builder, not before.

Variables per genset g, per step t:
    on[g,t]     binary   -- committed
    start[g,t]  binary   -- off-to-on transition (priced; also what min-up-time gates)
    stop[g,t]   binary   -- on-to-off transition (what min-down-time gates)
    output[g,t] continuous, `[min_stable_kw, rated_kw]` when on, 0 otherwise

Variables per battery s, per step t:
    charge[s,t], discharge[s,t]  continuous, >= 0
    is_charging[s,t]             binary, prevents simultaneous charge+discharge
                                  (without it, the solver could "arbitrage" a
                                  round-trip efficiency < 1 by charging and
                                  discharging in the same step, which no real
                                  battery can do)
    soc[s,t]                     continuous, `t = 0..T` (T+1 points: the
                                  initial SOC plus one after every step)

Station-level, per step t:
    renewable_used[t]  continuous, `[0, pv_available[t] + wind_available[t]]`
    boiler_heat[t]      continuous, `[0, boiler_rated_kw]`

Hard constraints (never traded against the objective, matching this
project's own "critical requirements are constraints, not priced" stance
carried over from `allotrope.safety`):
    - Electrical power balance is an equality: genset output + net battery
      discharge + renewable used == electrical_load_kw. Because
      electrical_load_kw already includes critical_load_kw, this is
      simultaneously the critical-load guarantee -- there is no separate
      "shed the discretionary part" decision in this formulation; see
      model.py's docstring for why (no unserved-load variable exists here
      at all, so there is nothing for the solver to trade fuel against).
    - Thermal service: CHP heat recovered + boiler heat >= firm_thermal_kw.
    - Battery SOC stays within `[soc_min, soc_max]` at every step.
    - Reserve margin: committed genset capacity (sum of rated_kw for every
      *on* unit) must exceed electrical_load_kw by at least
      `cfg.criticality.reserve_margin_kw` -- one defensible reading of
      "reserve" (N+1-style headroom against losing the marginal unit), not
      the only possible one; documented here rather than left implicit.

Objective: minimise total genset + boiler fuel cost, black-carbon cost, and
genset start cost, priced with the exact same `RewardWeights` the RL reward
uses (`fuel_per_l`, `black_carbon_per_g`, `genset_start_per_event`,
`curtailment_per_kwh`) -- not a second, independently-invented price list.
"""

from __future__ import annotations

import time

import numpy as np
import pulp

from allotrope.envs.reward import RewardWeights
from allotrope.optimization.model import LARGE_STEP_COUNT, MILPInstance
from allotrope.optimization.result import OptimizationResult


def solve(
    instance: MILPInstance,
    weights: RewardWeights | None = None,
    timeout_s: float | None = 60.0,
    msg: bool = False,
) -> OptimizationResult:
    """Solve one MILP instance. `timeout_s=None` runs CBC to proven
    optimality or infeasibility with no time limit -- use a real number for
    anything but a small offline benchmark, since CBC's worst case on a
    long horizon with many units is not fast. A timeout that expires with
    a feasible incumbent already found still returns that incumbent
    (status "Optimal" in PuLP's reporting is reserved for a *proven*
    optimum; an unproven-but-feasible timeout result reports PuLP's own
    "Not Solved"/"Undefined" status, and `OptimizationResult.feasible` is
    `False` for anything but a proven optimum -- callers that want to use
    an unproven-but-good incumbent must check `objective_value is not None`
    explicitly rather than trust `.feasible`."""
    weights = weights or RewardWeights()
    cfg = instance.cfg
    T = instance.horizon
    dt_h = instance.dt_h
    gensets = cfg.gensets
    storages = cfg.storage

    problem = pulp.LpProblem("allotrope_milp", pulp.LpMinimize)

    on = {
        (g.id, t): pulp.LpVariable(f"on_{g.id}_{t}", cat="Binary") for g in gensets for t in range(T)
    }
    start = {
        (g.id, t): pulp.LpVariable(f"start_{g.id}_{t}", cat="Binary")
        for g in gensets
        for t in range(T)
    }
    stop = {
        (g.id, t): pulp.LpVariable(f"stop_{g.id}_{t}", cat="Binary") for g in gensets for t in range(T)
    }
    output = {
        (g.id, t): pulp.LpVariable(f"output_{g.id}_{t}", lowBound=0, upBound=g.rated_kw)
        for g in gensets
        for t in range(T)
    }

    charge = {
        (s.id, t): pulp.LpVariable(f"charge_{s.id}_{t}", lowBound=0, upBound=s.max_charge_kw)
        for s in storages
        for t in range(T)
    }
    discharge = {
        (s.id, t): pulp.LpVariable(f"discharge_{s.id}_{t}", lowBound=0, upBound=s.max_discharge_kw)
        for s in storages
        for t in range(T)
    }
    is_charging = {
        (s.id, t): pulp.LpVariable(f"is_charging_{s.id}_{t}", cat="Binary")
        for s in storages
        for t in range(T)
    }
    soc = {
        (s.id, t): pulp.LpVariable(f"soc_{s.id}_{t}", lowBound=s.soc_min, upBound=s.soc_max)
        for s in storages
        for t in range(T + 1)
    }

    renewable_used = {
        t: pulp.LpVariable(
            f"renewable_used_{t}",
            lowBound=0,
            upBound=float(instance.pv_available_kw[t] + instance.wind_available_kw[t]),
        )
        for t in range(T)
    }
    boiler_heat = {
        t: pulp.LpVariable(f"boiler_heat_{t}", lowBound=0, upBound=cfg.thermal.boiler_rated_kw)
        for t in range(T)
    }

    # -- genset commitment: output bounds, transition bookkeeping, min up/down time --
    for g in gensets:
        up_steps = max(1, round(g.min_up_time_min / 60.0 / dt_h))
        down_steps = max(1, round(g.min_down_time_min / 60.0 / dt_h))
        prev_online = instance.initial_genset_online.get(g.id, False)
        for t in range(T):
            problem += output[(g.id, t)] >= g.min_stable_kw * on[(g.id, t)]
            problem += output[(g.id, t)] <= g.rated_kw * on[(g.id, t)]

            prior_on = on[(g.id, t - 1)] if t > 0 else (1 if prev_online else 0)
            problem += on[(g.id, t)] - prior_on == start[(g.id, t)] - stop[(g.id, t)]
            problem += start[(g.id, t)] + stop[(g.id, t)] <= 1

            up_window = [start[(g.id, i)] for i in range(max(0, t - up_steps + 1), t + 1)]
            problem += pulp.lpSum(up_window) <= on[(g.id, t)]
            down_window = [stop[(g.id, i)] for i in range(max(0, t - down_steps + 1), t + 1)]
            problem += pulp.lpSum(down_window) <= 1 - on[(g.id, t)]

    # -- battery SOC dynamics and charge/discharge exclusivity --
    for s in storages:
        problem += soc[(s.id, 0)] == instance.initial_soc.get(s.id, 0.5)
        for t in range(T):
            problem += charge[(s.id, t)] <= s.max_charge_kw * is_charging[(s.id, t)]
            problem += discharge[(s.id, t)] <= s.max_discharge_kw * (1 - is_charging[(s.id, t)])
            energy_in_kwh = charge[(s.id, t)] * s.one_way_efficiency * dt_h
            energy_out_kwh = discharge[(s.id, t)] / s.one_way_efficiency * dt_h
            problem += soc[(s.id, t + 1)] == soc[(s.id, t)] + (
                energy_in_kwh - energy_out_kwh
            ) / s.capacity_kwh

    # -- electrical power balance (hard: this is the critical-load guarantee) --
    for t in range(T):
        genset_total = pulp.lpSum(output[(g.id, t)] for g in gensets)
        battery_net = pulp.lpSum(discharge[(s.id, t)] - charge[(s.id, t)] for s in storages)
        problem += (
            genset_total + battery_net + renewable_used[t]
            == float(instance.electrical_load_kw[t])
        )
        # Reserve margin: committed capacity must exceed load by the
        # station's configured margin -- an N+1-style headroom reading of
        # "reserve," not the only possible one (see module docstring).
        committed_capacity = pulp.lpSum(g.rated_kw * on[(g.id, t)] for g in gensets)
        problem += committed_capacity >= float(instance.electrical_load_kw[t]) + cfg.criticality.reserve_margin_kw

    # -- thermal service (hard) --
    for t in range(T):
        recovered_heat = pulp.lpSum(output[(g.id, t)] * g.chp_heat_ratio for g in gensets)
        problem += recovered_heat + boiler_heat[t] >= float(instance.firm_thermal_kw[t])

    # -- objective: fuel + black carbon + starts + curtailment, RewardWeights' own prices --
    lhv_kwh_per_l = gensets[0].fuel_lhv_mj_per_l / 3.6
    fuel_terms = []
    bc_terms = []
    for g in gensets:
        for t in range(T):
            fuel_l_t = dt_h * (g.willans_intercept_l_per_h * on[(g.id, t)] + g.willans_slope_l_per_kwh * output[(g.id, t)])
            fuel_terms.append(fuel_l_t)
            bc_terms.append(dt_h * g.bc_ef_clean_mg_per_kwh * output[(g.id, t)] / 1000.0)
    boiler_fuel_terms = [
        dt_h * boiler_heat[t] / (lhv_kwh_per_l * cfg.thermal.boiler_efficiency) for t in range(T)
    ]
    start_terms = [start[(g.id, t)] for g in gensets for t in range(T)]
    curtail_terms = [
        (float(instance.pv_available_kw[t] + instance.wind_available_kw[t]) - renewable_used[t]) * dt_h
        for t in range(T)
    ]

    problem += (
        weights.fuel_per_l * (pulp.lpSum(fuel_terms) + pulp.lpSum(boiler_fuel_terms))
        + weights.black_carbon_per_g * pulp.lpSum(bc_terms)
        + weights.genset_start_per_event * pulp.lpSum(start_terms)
        + weights.curtailment_per_kwh * pulp.lpSum(curtail_terms)
    )

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=timeout_s)
    start_time = time.perf_counter()
    problem.solve(solver)
    solve_time_s = time.perf_counter() - start_time

    status = pulp.LpStatus[problem.status]
    n_g, n_s = len(gensets), len(storages)

    def _val(var: pulp.LpVariable) -> float:
        v = var.value()
        return float(v) if v is not None else 0.0

    genset_on_arr = np.array([[round(_val(on[(g.id, t)])) for g in gensets] for t in range(T)])
    genset_output_arr = np.array([[_val(output[(g.id, t)]) for g in gensets] for t in range(T)])
    genset_started_arr = np.array([[round(_val(start[(g.id, t)])) for g in gensets] for t in range(T)])
    battery_charge_arr = np.array([[_val(charge[(s.id, t)]) for s in storages] for t in range(T)]) if n_s else np.zeros((T, 0))
    battery_discharge_arr = np.array([[_val(discharge[(s.id, t)]) for s in storages] for t in range(T)]) if n_s else np.zeros((T, 0))
    battery_soc_arr = np.array([[_val(soc[(s.id, t)]) for s in storages] for t in range(T + 1)]) if n_s else np.zeros((T + 1, 0))
    boiler_heat_arr = np.array([_val(boiler_heat[t]) for t in range(T)])
    renewable_used_arr = np.array([_val(renewable_used[t]) for t in range(T)])
    curtailed_arr = (instance.pv_available_kw + instance.wind_available_kw) - renewable_used_arr

    fuel_l_arr = np.array(
        [
            sum(
                dt_h
                * (
                    g.willans_intercept_l_per_h * genset_on_arr[t, k]
                    + g.willans_slope_l_per_kwh * genset_output_arr[t, k]
                )
                for k, g in enumerate(gensets)
            )
            + dt_h * boiler_heat_arr[t] / (lhv_kwh_per_l * cfg.thermal.boiler_efficiency)
            for t in range(T)
        ]
    )
    black_carbon_arr = np.array(
        [
            sum(
                dt_h * g.bc_ef_clean_mg_per_kwh * genset_output_arr[t, k]
                for k, g in enumerate(gensets)
            )
            for t in range(T)
        ]
    )

    return OptimizationResult(
        status=status,
        objective_value=pulp.value(problem.objective),
        solve_time_s=solve_time_s,
        dt_h=dt_h,
        genset_ids=[g.id for g in gensets],
        storage_ids=[s.id for s in storages],
        genset_on=genset_on_arr,
        genset_output_kw=genset_output_arr,
        genset_started=genset_started_arr,
        battery_charge_kw=battery_charge_arr,
        battery_discharge_kw=battery_discharge_arr,
        battery_soc=battery_soc_arr,
        boiler_heat_kw=boiler_heat_arr,
        renewable_used_kw=renewable_used_arr,
        curtailed_kw=curtailed_arr,
        fuel_l=fuel_l_arr,
        black_carbon_g=black_carbon_arr,
    )


__all__ = ["solve", "LARGE_STEP_COUNT"]
