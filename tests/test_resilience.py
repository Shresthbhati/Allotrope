"""Correctness tests for the resilience benchmark.

These check the benchmark's own plumbing -- that faults actually reach the
plant when driven through `run_episode` (not just through the Gymnasium
wrapper `test_faults.py` already covers), that scenarios are well-formed,
and that results are deterministic and structured as documented. They do
not assert *which* controller is "more resilient" -- that would be a
claim about controller quality, not about whether this tool works.
"""

from __future__ import annotations

import pytest

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.resilience import STANDARD_SCENARIOS, run_benchmark, run_scenario
from allotrope.resilience.scenarios import (
    battery_lockout_at_peak,
    compound_failure,
    extended_whiteout,
    single_genset_trip,
)


@pytest.fixture(scope="module")
def cfg():
    return load_station("maitri")


def test_every_standard_scenario_builds_a_valid_schedule_for_a_real_station(cfg):
    genset_ids = {g.id for g in cfg.gensets}
    storage_ids = {s.id for s in cfg.storage}
    for name, make_schedule in STANDARD_SCENARIOS.items():
        schedule = make_schedule(cfg)
        for event in schedule:
            if event.kind == "genset_trip":
                assert event.target_id in genset_ids, name
            if event.kind == "battery_lockout":
                assert event.target_id in storage_ids, name


def test_a_genset_trip_scenario_actually_takes_a_genset_offline(cfg):
    schedule = single_genset_trip(cfg)
    trip = schedule[0]
    controller = LegacyNPlusOne(cfg)
    result = run_scenario(controller, cfg, schedule, periods=168, seed=1)

    online_col = f"genset_online_{[g.id for g in cfg.gensets].index(trip.target_id)}"
    trace = result.telemetry.reset_index()
    during_fault = trace.iloc[trip.start_step : trip.end_step]
    assert not during_fault[online_col].any(), "tripped genset reported online during its fault window"


def test_a_battery_lockout_scenario_zeroes_that_batterys_power(cfg):
    schedule = battery_lockout_at_peak(cfg)
    lockout = schedule[0]
    controller = EfficientRuleBased(cfg)
    result = run_scenario(controller, cfg, schedule, periods=168, seed=1)

    b_index = [s.id for s in cfg.storage].index(lockout.target_id)
    col = f"battery_kw_{b_index}"
    during_fault = result.telemetry.iloc[lockout.start_step : lockout.end_step]
    assert (during_fault[col].abs() < 1e-9).all()


def test_extended_whiteout_actually_reduces_renewable_availability(cfg):
    schedule = extended_whiteout(cfg)
    derate = schedule[0]
    controller = EfficientRuleBased(cfg)

    with_fault = run_scenario(controller, cfg, schedule, periods=168, seed=1)
    without_fault = run_scenario(controller, cfg, [], periods=168, seed=1)

    during = slice(derate.start_step, derate.end_step)
    faulted_available = (
        with_fault.telemetry["pv_available_kw"].iloc[during]
        + with_fault.telemetry["wind_available_kw"].iloc[during]
    )
    baseline_available = (
        without_fault.telemetry["pv_available_kw"].iloc[during]
        + without_fault.telemetry["wind_available_kw"].iloc[during]
    )
    assert faulted_available.sum() == pytest.approx(baseline_available.sum() * 0.15, rel=1e-6)


def test_compound_failure_combines_a_derate_and_a_trip(cfg):
    schedule = compound_failure(cfg)
    kinds = {e.kind for e in schedule}
    assert kinds == {"renewable_derate", "genset_trip"}


def test_run_benchmark_produces_one_row_per_controller_and_scenario(cfg):
    controllers = {"legacy": lambda c: LegacyNPlusOne(c), "efficient": lambda c: EfficientRuleBased(c)}
    scenarios = {"baseline": STANDARD_SCENARIOS["baseline"], "single_genset_trip": single_genset_trip}
    df = run_benchmark(controllers, cfg, scenarios=scenarios, periods=72, seed=1)

    assert len(df) == len(controllers) * len(scenarios)
    assert set(df["controller"]) == set(controllers)
    assert set(df["scenario"]) == set(scenarios)
    for key in ("fuel_l", "critical_unserved_kwh", "genset_starts"):
        assert key in df.columns


def test_the_baseline_scenario_matches_an_unfaulted_run(cfg):
    controller = EfficientRuleBased(cfg)
    faulted = run_scenario(controller, cfg, STANDARD_SCENARIOS["baseline"](cfg), periods=72, seed=1)
    unfaulted = run_scenario(EfficientRuleBased(cfg), cfg, [], periods=72, seed=1)
    assert faulted.summary == unfaulted.summary


def test_running_the_same_scenario_twice_is_deterministic(cfg):
    schedule = compound_failure(cfg)
    r1 = run_scenario(EfficientRuleBased(cfg), cfg, schedule, periods=96, seed=2)
    r2 = run_scenario(EfficientRuleBased(cfg), cfg, schedule, periods=96, seed=2)
    assert r1.summary == r2.summary
