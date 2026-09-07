"""Correctness tests for the scripted equipment-failure injector.

Each test drives the plant with an action that actively tries to defeat the
fault (start the tripped genset, charge/discharge the locked battery) to
prove the fault wins regardless of what the controller wants -- a fault
that only holds against a passive controller would not be testing much.
"""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.faults import FaultEvent, FaultInjector


@pytest.fixture
def env():
    return PolarMicrogridEnv(
        station="maitri", start="2026-06-01", periods=96, seed=3, episode_steps=96
    )


def _force_everything_on(env):
    action = env.action_space.sample()
    action["genset_on"][:] = 1
    action["dispatch"][:] = 1.0  # max discharge/output request everywhere
    return action


def test_a_tripped_genset_cannot_be_started_by_any_action(env):
    g0 = env.cfg.gensets[0].id
    schedule = [FaultEvent(kind="genset_trip", start_step=5, duration_steps=10, target_id=g0)]
    wrapped = FaultInjector(env, schedule)
    wrapped.reset(seed=3)

    for t in range(30):
        _, _, terminated, truncated, info = wrapped.step(_force_everything_on(env))
        online = dict(zip([g.id for g in env.cfg.gensets], info["telemetry"]["genset_online"]))
        if 5 <= t < 15:
            assert online[g0] is False, f"genset tripped but reported online at step {t}"
            assert "genset_trip" in info["active_faults"]
        assert not terminated and not truncated


def test_a_tripped_genset_can_restart_immediately_after_the_fault_ends(env):
    g0 = env.cfg.gensets[0].id
    schedule = [FaultEvent(kind="genset_trip", start_step=5, duration_steps=10, target_id=g0)]
    wrapped = FaultInjector(env, schedule)
    wrapped.reset(seed=3)

    for t in range(20):
        _, _, _, _, info = wrapped.step(_force_everything_on(env))

    online = dict(zip([g.id for g in env.cfg.gensets], info["telemetry"]["genset_online"]))
    assert online[g0] is True
    assert "genset_trip" not in info["active_faults"]


def test_a_locked_out_battery_delivers_exactly_zero_power(env):
    b0 = env.cfg.storage[0].id
    schedule = [FaultEvent(kind="battery_lockout", start_step=4, duration_steps=8, target_id=b0)]
    wrapped = FaultInjector(env, schedule)
    wrapped.reset(seed=3)

    b0_index = [s.id for s in env.cfg.storage].index(b0)
    for t in range(20):
        _, _, _, _, info = wrapped.step(_force_everything_on(env))
        if 4 <= t < 12:
            assert info["telemetry"]["battery_kw"][b0_index] == pytest.approx(0.0, abs=1e-9)
            assert "battery_lockout" in info["active_faults"]


def test_renewable_derate_scales_available_power_only_inside_the_window(env):
    schedule = [FaultEvent(kind="renewable_derate", start_step=10, duration_steps=5, magnitude=0.7)]
    wrapped = FaultInjector(env, schedule)
    wrapped.reset(seed=3)

    for t in range(20):
        _, _, _, _, info = wrapped.step(env.action_space.sample())
        tel = info["telemetry"]
        pristine_pv = wrapped._pristine_pv[env.plant.state.step_index - 1]
        pristine_wind = wrapped._pristine_wind[env.plant.state.step_index - 1]
        if 10 <= t < 15:
            assert tel["pv_available_kw"] == pytest.approx(pristine_pv * 0.3, rel=1e-6)
            assert tel["wind_available_kw"] == pytest.approx(pristine_wind * 0.3, rel=1e-6)
        else:
            assert tel["pv_available_kw"] == pytest.approx(pristine_pv, rel=1e-6)
            assert tel["wind_available_kw"] == pytest.approx(pristine_wind, rel=1e-6)


def test_replaying_the_same_schedule_is_deterministic(env):
    g0 = env.cfg.gensets[0].id
    b0 = env.cfg.storage[0].id
    schedule = [
        FaultEvent(kind="genset_trip", start_step=3, duration_steps=6, target_id=g0),
        FaultEvent(kind="battery_lockout", start_step=5, duration_steps=4, target_id=b0),
        FaultEvent(kind="renewable_derate", start_step=0, duration_steps=20, magnitude=0.5),
    ]

    def run():
        e = PolarMicrogridEnv(
            station="maitri", start="2026-06-01", periods=96, seed=3, episode_steps=96
        )
        wrapped = FaultInjector(e, schedule)
        wrapped.reset(seed=3)
        e.action_space.seed(3)
        history = []
        for _ in range(20):
            _, reward, _, _, info = wrapped.step(e.action_space.sample())
            history.append((reward, tuple(info["telemetry"]["genset_online"])))
        return history

    assert run() == run()


def test_fault_event_validates_its_own_fields():
    with pytest.raises(ValueError):
        FaultEvent(kind="genset_trip", start_step=0, duration_steps=1, target_id=None)
    with pytest.raises(ValueError):
        FaultEvent(kind="renewable_derate", start_step=0, duration_steps=1, target_id="G1")
    with pytest.raises(ValueError):
        FaultEvent(kind="renewable_derate", start_step=0, duration_steps=1, magnitude=0.0)
    with pytest.raises(ValueError):
        FaultEvent(kind="genset_trip", start_step=0, duration_steps=0, target_id="G1")
    with pytest.raises(ValueError):
        FaultEvent(kind="genset_trip", start_step=-1, duration_steps=1, target_id="G1")


def test_the_injector_rejects_an_unknown_target_id(env):
    with pytest.raises(ValueError):
        FaultInjector(
            env, [FaultEvent(kind="genset_trip", start_step=0, duration_steps=1, target_id="nope")]
        )


def test_no_active_faults_reproduces_the_unwrapped_plants_behaviour(env):
    """An injector with an empty schedule must not change anything -- the
    physical overrides it applies are keyed on active faults, so an empty
    schedule should leave the plant byte-for-byte the same as running it
    directly."""
    baseline = PolarMicrogridEnv(
        station="maitri", start="2026-06-01", periods=96, seed=3, episode_steps=96
    )
    baseline.reset(seed=3)
    baseline.action_space.seed(3)

    wrapped_env = PolarMicrogridEnv(
        station="maitri", start="2026-06-01", periods=96, seed=3, episode_steps=96
    )
    wrapped = FaultInjector(wrapped_env, [])
    wrapped.reset(seed=3)
    wrapped_env.action_space.seed(3)

    for _ in range(20):
        action = baseline.action_space.sample()
        _, r1, _, _, i1 = baseline.step(action)
        _, r2, _, _, i2 = wrapped.step(action)
        assert r1 == pytest.approx(r2)
        assert i1["telemetry"]["genset_online"] == i2["telemetry"]["genset_online"]
        assert np.allclose(i1["telemetry"]["battery_kw"], i2["telemetry"]["battery_kw"])
