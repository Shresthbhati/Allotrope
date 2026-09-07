"""Correctness tests for the offline MILP oracle.

Each test targets one property the module's own docstrings claim: the
electrical balance is a hard equality (so a solve *is* the critical-load
guarantee), SOC/output stay in bounds, commitment bookkeeping is honest,
solves are deterministic, and a starved timeout returns gracefully instead
of raising.
"""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.optimization import instance_from_plant, solve
from allotrope.sim.runner import build_plant


@pytest.fixture(scope="module")
def cfg():
    return load_station("maitri")


@pytest.fixture
def instance(cfg):
    plant = build_plant(cfg, start="2026-06-01", periods=48, seed=1)
    plant.reset()
    controller = EfficientRuleBased(cfg)
    for _ in range(5):
        plant.step(controller.act(plant.observe(), plant))
    return instance_from_plant(plant, start_index=5, horizon=24)


def test_a_realistic_instance_solves_to_a_proven_optimum(instance):
    result = solve(instance, timeout_s=30.0)
    assert result.feasible
    assert result.status == "Optimal"
    assert result.objective_value is not None


def test_electrical_load_is_served_exactly_every_step(instance):
    """The MILP has no unserved-load variable, so this is the critical-load guarantee."""
    result = solve(instance, timeout_s=30.0)
    battery_net = result.battery_discharge_kw.sum(axis=1) - result.battery_charge_kw.sum(axis=1)
    supply = result.genset_output_kw.sum(axis=1) + battery_net + result.renewable_used_kw
    assert supply == pytest.approx(instance.electrical_load_kw, abs=1e-4)


def test_firm_thermal_load_is_met_every_step(instance, cfg):
    result = solve(instance, timeout_s=30.0)
    recovered = sum(
        result.genset_output_kw[:, k] * g.chp_heat_ratio for k, g in enumerate(cfg.gensets)
    )
    assert (recovered + result.boiler_heat_kw + 1e-6 >= instance.firm_thermal_kw).all()


def test_battery_soc_stays_within_configured_bounds(instance, cfg):
    result = solve(instance, timeout_s=30.0)
    for k, s in enumerate(cfg.storage):
        assert (result.battery_soc[:, k] >= s.soc_min - 1e-6).all()
        assert (result.battery_soc[:, k] <= s.soc_max + 1e-6).all()


def test_genset_output_respects_min_stable_and_rated_bounds(instance, cfg):
    result = solve(instance, timeout_s=30.0)
    for k, g in enumerate(cfg.gensets):
        online = result.genset_on[:, k].astype(bool)
        assert np.allclose(result.genset_output_kw[~online, k], 0.0, atol=1e-6)
        assert (result.genset_output_kw[online, k] >= g.min_stable_kw - 1e-6).all()
        assert (result.genset_output_kw[:, k] <= g.rated_kw + 1e-6).all()


def test_start_stop_bookkeeping_matches_the_on_trajectory(instance, cfg):
    result = solve(instance, timeout_s=30.0)
    on, started = result.genset_on, result.genset_started
    for k, g in enumerate(cfg.gensets):
        prev = 1 if instance.initial_genset_online.get(g.id, False) else 0
        for t in range(on.shape[0]):
            assert started[t, k] == max(on[t, k] - prev, 0)
            prev = on[t, k]


def test_reserve_margin_is_respected_every_step(instance, cfg):
    result = solve(instance, timeout_s=30.0)
    committed_capacity = sum(
        g.rated_kw * result.genset_on[:, k] for k, g in enumerate(cfg.gensets)
    )
    assert (
        committed_capacity + 1e-6 >= instance.electrical_load_kw + cfg.criticality.reserve_margin_kw
    ).all()


def test_solving_the_same_instance_twice_gives_the_same_schedule(instance):
    r1 = solve(instance, timeout_s=30.0)
    r2 = solve(instance, timeout_s=30.0)
    assert r1.objective_value == pytest.approx(r2.objective_value, abs=1e-6)
    assert np.array_equal(r1.genset_on, r2.genset_on)


def test_a_very_short_timeout_returns_a_result_instead_of_raising(instance):
    result = solve(instance, timeout_s=0.01)
    assert result.status in ("Optimal", "Not Solved", "Undefined")


def test_summary_omits_metrics_the_milp_cannot_produce(instance):
    result = solve(instance, timeout_s=30.0)
    summary = result.summary()
    assert "wet_stacking_fraction" not in summary
    assert "mean_deposit" not in summary
    assert summary["fuel_l"] >= 0.0
