"""Correctness tests for per-sensor staleness/plausibility classification.

`specs_for_station` ships with staleness checking off for every field (see
its docstring: this project's noiseless synthetic telemetry has long,
legitimate flat stretches -- polar night, a pinned battery envelope, a
flat baseload -- that a naive repeat-count would misclassify). These tests
therefore check the mechanism itself (`classify`, `SensorHistory`,
`ObservationHealth`) directly with specs that turn staleness on, plus a
real-run check that the shipped defaults produce no false positives.
"""

from __future__ import annotations

import pytest

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.safety.sensor_health import (
    ObservationHealth,
    SensorHistory,
    SensorSpec,
    SensorStatus,
    classify,
    specs_for_station,
    worse,
)
from allotrope.sim.runner import build_plant


def test_a_normal_value_in_range_is_valid():
    spec = SensorSpec("x", 0.0, 100.0, max_stale_steps=0)
    history = SensorHistory()
    assert classify(50.0, spec, history) == SensorStatus.VALID


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_corrupted(bad):
    spec = SensorSpec("x", 0.0, 100.0, max_stale_steps=0)
    assert classify(bad, spec, SensorHistory()) == SensorStatus.CORRUPTED


def test_an_out_of_range_but_finite_value_is_corrupted():
    spec = SensorSpec("soc", 0.0, 1.0, max_stale_steps=0)
    assert classify(5.0, spec, SensorHistory()) == SensorStatus.CORRUPTED
    assert classify(-0.5, spec, SensorHistory()) == SensorStatus.CORRUPTED


def test_a_non_numeric_value_is_corrupted():
    spec = SensorSpec("x", 0.0, 100.0, max_stale_steps=0)
    assert classify("not a number", spec, SensorHistory()) == SensorStatus.CORRUPTED
    assert classify(None, spec, SensorHistory()) == SensorStatus.CORRUPTED


def test_a_value_repeated_past_the_threshold_is_stale():
    spec = SensorSpec("x", 0.0, 100.0, max_stale_steps=3)
    history = SensorHistory()
    statuses = [classify(42.0, spec, history) for _ in range(5)]
    # First reading: nothing to compare against yet, so VALID. Then three
    # more identical readings before the threshold trips at the fourth.
    assert statuses == [
        SensorStatus.VALID,
        SensorStatus.VALID,
        SensorStatus.VALID,
        SensorStatus.STALE,
        SensorStatus.STALE,
    ]


def test_a_changing_value_never_goes_stale():
    spec = SensorSpec("x", 0.0, 100.0, max_stale_steps=1)
    history = SensorHistory()
    statuses = [classify(float(i), spec, history) for i in range(10)]
    assert all(s == SensorStatus.VALID for s in statuses)


def test_zero_max_stale_steps_disables_staleness_checking():
    spec = SensorSpec("x", 0.0, 100.0, max_stale_steps=0)
    history = SensorHistory()
    statuses = [classify(42.0, spec, history) for _ in range(50)]
    assert all(s == SensorStatus.VALID for s in statuses)


def test_worse_orders_statuses_correctly():
    assert worse(SensorStatus.VALID, SensorStatus.STALE) == SensorStatus.STALE
    assert worse(SensorStatus.CORRUPTED, SensorStatus.STALE) == SensorStatus.CORRUPTED
    assert worse(SensorStatus.VALID, SensorStatus.VALID) == SensorStatus.VALID


def test_observation_health_grades_an_array_field_by_its_worst_element():
    spec = SensorSpec("battery_soc", 0.0, 1.0, max_stale_steps=0)
    health = ObservationHealth({"battery_soc": spec})
    statuses = health.check({"battery_soc": [0.5, 5.0, 0.3]})
    assert statuses["battery_soc"] == SensorStatus.CORRUPTED


def test_observation_health_flags_a_missing_field_as_corrupted():
    spec = SensorSpec("x", 0.0, 100.0, max_stale_steps=0)
    health = ObservationHealth({"x": spec})
    statuses = health.check({})
    assert statuses["x"] == SensorStatus.CORRUPTED


def test_the_shipped_station_specs_produce_no_false_positives_on_a_real_run():
    """The whole reason max_stale_steps defaults to 0 for every field
    (see specs_for_station's docstring): this project's synthetic
    telemetry has long, legitimate flat stretches that a naive
    repeat-count would otherwise misclassify. This is the regression test
    for that finding."""
    cfg = load_station("maitri")
    plant = build_plant(cfg, start="2026-06-01", periods=336, seed=1)
    plant.reset()
    controller = EfficientRuleBased(cfg)
    health = ObservationHealth(specs_for_station(cfg))

    statuses_seen: set[SensorStatus] = set()
    for _ in range(336):
        obs = plant.observe()
        statuses_seen.update(health.check(obs).values())
        plant.step(controller.act(obs, plant))

    assert statuses_seen == {SensorStatus.VALID}


def test_the_shipped_station_specs_still_catch_real_corruption():
    cfg = load_station("maitri")
    plant = build_plant(cfg, start="2026-06-01", periods=24, seed=1)
    plant.reset()
    health = ObservationHealth(specs_for_station(cfg))
    obs = plant.observe()

    obs["critical_load_kw"] = float("nan")
    obs["battery_soc"] = [5.0] * len(cfg.storage)
    obs["pv_available_kw"] = -10.0

    statuses = health.check(obs)
    assert statuses["critical_load_kw"] == SensorStatus.CORRUPTED
    assert statuses["battery_soc"] == SensorStatus.CORRUPTED
    assert statuses["pv_available_kw"] == SensorStatus.CORRUPTED


def test_worst_reports_the_single_worst_status_across_all_fields():
    specs = specs_for_station(load_station("maitri"))
    health = ObservationHealth(specs)
    good = {k: (0.0 if not k.endswith("_soc") else [0.5] * 2) for k in specs}
    assert health.worst(good) == SensorStatus.VALID

    bad = dict(good)
    bad["critical_load_kw"] = float("nan")
    assert health.worst(bad) == SensorStatus.CORRUPTED
