"""The Gymnasium environment and the reward it optimises."""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.config import available_stations, load_station
from allotrope.control.baseline import EfficientRuleBased, LegacyNPlusOne
from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.envs.reward import RewardFunction, RewardWeights

WINTER = "2026-06-01"


@pytest.fixture
def env():
    return PolarMicrogridEnv(station="maitri", start=WINTER, periods=24 * 7, seed=0)


# -- gymnasium conformance ----------------------------------------------------


def test_reset_returns_an_observation_inside_the_declared_space(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_returns_the_five_tuple_gymnasium_expects(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "telemetry" in info and "reward_breakdown" in info


def test_observations_stay_finite_and_bounded_across_a_full_episode(env):
    env.reset(seed=0)
    env.action_space.seed(0)
    while True:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break


def test_the_episode_terminates_at_the_end_of_the_weather(env):
    env.reset(seed=0)
    steps = 0
    while steps < env.plant.n_steps + 5:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert terminated or truncated
    assert steps <= env.plant.n_steps


def test_episode_length_is_respected_when_requested():
    env = PolarMicrogridEnv(
        station="maitri", start=WINTER, periods=24 * 30, seed=0, episode_steps=48
    )
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps == 48 and truncated


def test_the_same_seed_reproduces_the_same_episode(env):
    def rollout(seed):
        e = PolarMicrogridEnv(station="maitri", start=WINTER, periods=24 * 3, seed=seed)
        e.reset(seed=seed)
        e.action_space.seed(0)
        rewards = []
        while True:
            _, r, term, trunc, _ = e.step(e.action_space.sample())
            rewards.append(r)
            if term or trunc:
                break
        return rewards

    assert rollout(4) == rollout(4)
    assert rollout(4) != rollout(5)


def test_reset_clears_the_previous_episode(env):
    env.reset(seed=0)
    for _ in range(24):
        env.step(env.action_space.sample())
    assert env.summary()["fuel_l"] > 0.0

    env.reset(seed=0)
    assert env.summary()["fuel_l"] == 0.0


@pytest.mark.parametrize("station", available_stations())
def test_observation_width_matches_the_declared_space(station):
    env = PolarMicrogridEnv(station=station, periods=48, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_forecast_features_fall_back_to_the_current_reading_before_any_history():
    """No steps yet -> no honest forecast yet -> use the instantaneous value.

    Matches PersistenceForecaster's and EWMAForecaster's own behaviour on a
    single-point history: the best available estimate of the future, before
    any history exists, is simply "now" -- not a fabricated number.
    """
    env = PolarMicrogridEnv(station="maitri", start=WINTER, periods=48, seed=0)
    obs, _ = env.reset(seed=0)
    load_idx, renewable_idx = 12, 13
    scale = env._power_scale_kw
    raw = env.plant.observe()
    assert obs[load_idx] == pytest.approx(raw["electrical_load_kw"] / scale, abs=1e-5)
    assert obs[renewable_idx] == pytest.approx(
        (raw["pv_available_kw"] + raw["wind_available_kw"]) / scale, abs=1e-5
    )


def test_forecast_features_match_an_independently_computed_ewma():
    """After real history exists, the observation's forecast must equal
    EWMAForecaster applied to that same history -- not merely "some number"."""
    from allotrope.intelligence.forecasting import EWMAForecaster

    env = PolarMicrogridEnv(station="maitri", start=WINTER, periods=48, seed=0)
    obs, _ = env.reset(seed=0)
    action = {
        "genset_on": np.zeros(len(env.cfg.gensets), dtype=np.int8),
        "dispatch": np.zeros(len(env.cfg.gensets) + len(env.cfg.storage) + 1, dtype=np.float32),
    }
    for _ in range(30):
        obs, _, _, _, _ = env.step(action)

    scale = env._power_scale_kw
    expected_load = EWMAForecaster().forecast(np.asarray(env._load_history), 24)
    expected_renewable = EWMAForecaster().forecast(np.asarray(env._renewable_history), 24)
    assert obs[12] == pytest.approx(np.clip(expected_load / scale, -5.0, 5.0), abs=1e-4)
    assert obs[13] == pytest.approx(np.clip(expected_renewable / scale, -5.0, 5.0), abs=1e-4)


def test_wear_score_and_fec_features_track_the_asset_health_tracker():
    """Not a second bookkeeping system: the observation's wear/FEC features
    must equal exactly what allotrope.intelligence.asset_health computes,
    scaled the documented way -- run for real against EfficientRuleBased so
    at least one genset actually starts and accrues real wear."""
    from allotrope.control.baseline import EfficientRuleBased

    env = PolarMicrogridEnv(station="maitri", start=WINTER, periods=24 * 3, seed=0)
    controller = EfficientRuleBased(cfg=env.cfg)
    obs, _ = env.reset(seed=0)

    n_g, n_s = len(env.cfg.gensets), len(env.cfg.storage)
    wear_start = 14 + 5 * n_g + 2 * n_s
    fec_start = 14 + 6 * n_g + 2 * n_s

    # Stop one step short of the episode's natural end: step() intentionally
    # returns an all-zero observation on termination (there is no
    # meaningful "next state" past the end of the weather), which would
    # make this test compare against that zero vector instead of a real one.
    for _ in range(24 * 3 - 1):
        command = controller.act(env.plant.observe(), env.plant)
        obs, _, terminated, truncated, _ = env.step(env.encode_command(command))
        assert not terminated and not truncated

    total_starts = sum(h.starts for h in env._asset_tracker.gensets.values())
    assert total_starts > 0, "test is meaningless if no genset ever started"

    for k, g in enumerate(env.cfg.gensets):
        expected = np.clip(env._asset_tracker.gensets[g.id].wear_score().value / 5000.0, -5.0, 5.0)
        assert obs[wear_start + k] == pytest.approx(expected, abs=1e-4)
    for k, s in enumerate(env.cfg.storage):
        expected = np.clip(
            env._asset_tracker.batteries[s.id].full_equivalent_cycles().value, -5.0, 5.0
        )
        assert obs[fec_start + k] == pytest.approx(expected, abs=1e-4)


def test_stations_of_different_sizes_produce_comparable_observations():
    """Scaling by installed capacity is what lets one policy serve both stations."""
    small = PolarMicrogridEnv(station="maitri", start=WINTER, periods=48, seed=0)
    large = PolarMicrogridEnv(station="bharati", start=WINTER, periods=48, seed=0)
    assert large.cfg.total_genset_kw > small.cfg.total_genset_kw * 1.5

    obs_small, _ = small.reset(seed=0)
    obs_large, _ = large.reset(seed=0)
    # The load feature is index 0 in both cases and should be the same order.
    assert abs(obs_small[0] - obs_large[0]) < 0.25


# -- action encoding ----------------------------------------------------------


def test_decoded_setpoints_land_inside_each_machine_band(env):
    env.reset(seed=0)
    env.action_space.seed(1)
    for _ in range(50):
        command = env.decode_action(env.action_space.sample())
        for k, g in enumerate(env.cfg.gensets):
            assert g.min_stable_kw - 1e-6 <= command.genset_setpoint_kw[k] <= g.rated_kw + 1e-6
        assert command.snow_melt_kw >= 0.0


def test_encode_and_decode_are_inverse_for_rule_based_commands(env):
    """Rule-based and learned controllers must be scorable on the same path."""
    env.reset(seed=0)
    controller = EfficientRuleBased(env.cfg)
    for _ in range(48):
        original = controller.act(env.plant.observe(), env.plant)
        recovered = env.decode_action(env.encode_command(original))

        assert recovered.genset_on == original.genset_on
        for k in range(len(env.cfg.gensets)):
            if original.genset_on[k]:
                assert recovered.genset_setpoint_kw[k] == pytest.approx(
                    original.genset_setpoint_kw[k], rel=1e-4, abs=1e-4
                )
        assert recovered.snow_melt_kw == pytest.approx(original.snow_melt_kw, rel=1e-4, abs=1e-4)
        env.step(env.encode_command(original))


# -- reward -------------------------------------------------------------------


def _telemetry(**overrides):
    base = {
        "fuel_l": 0.0,
        "black_carbon_mg": 0.0,
        "genset_starts": 0,
        "curtailed_kw": 0.0,
        "unserved_kw": 0.0,
        "critical_unserved_kw": 0.0,
        "indoor_temp_c": 20.0,
        "min_indoor_temp_c": 16.0,
    }
    base.update(overrides)
    return base


def test_burning_fuel_costs_reward():
    reward_fn = RewardFunction()
    idle, _ = reward_fn(_telemetry(), dt_h=1.0, deposit_delta=0.0)
    burning, _ = reward_fn(_telemetry(fuel_l=50.0), dt_h=1.0, deposit_delta=0.0)
    assert burning < idle


def test_life_support_dominates_any_reachable_fuel_saving():
    """The safety terms must not be tradeable, or the agent will trade them.

    The comparison is against the largest fuel saving physically available in one
    step -- shutting the entire plant down from full load -- rather than against
    an arbitrary quantity, because that is the whole of the temptation on offer.
    """
    cfg = load_station("maitri")
    reward_fn = RewardFunction()
    max_burn_l_per_h = sum(g.fuel_rate_rated_l_per_h for g in cfg.gensets)

    _, shed = reward_fn(_telemetry(critical_unserved_kw=1.0), dt_h=1.0, deposit_delta=0.0)
    _, burnt = reward_fn(_telemetry(fuel_l=max_burn_l_per_h), dt_h=1.0, deposit_delta=0.0)

    assert shed.critical_unserved > burnt.fuel * 10, (
        "one kWh of shed life support must cost far more than an hour at full load"
    )


def test_serving_the_station_always_beats_shedding_it():
    reward_fn = RewardFunction()
    _, saved = reward_fn(_telemetry(fuel_l=100.0), dt_h=1.0, deposit_delta=0.0)
    _, shed = reward_fn(_telemetry(critical_unserved_kw=1.0), dt_h=1.0, deposit_delta=0.0)
    assert saved.total_cost < shed.total_cost


def test_freezing_is_penalised_only_below_the_hard_floor():
    reward_fn = RewardFunction()
    _, cool = reward_fn(_telemetry(indoor_temp_c=17.0), dt_h=1.0, deposit_delta=0.0)
    _, frozen = reward_fn(_telemetry(indoor_temp_c=15.0), dt_h=1.0, deposit_delta=0.0)
    assert cool.freeze == 0.0
    assert frozen.freeze > 0.0


def test_machine_starts_are_priced():
    """The efficient baseline makes 307 starts a year; the reward must notice."""
    reward_fn = RewardFunction()
    _, still = reward_fn(_telemetry(), dt_h=1.0, deposit_delta=0.0)
    _, cycling = reward_fn(_telemetry(genset_starts=1), dt_h=1.0, deposit_delta=0.0)
    assert cycling.starts > 0.0
    assert cycling.total_cost > still.total_cost


def test_fouling_is_charged_but_cleaning_is_not_rewarded():
    """Paying for deposit reduction would let an agent farm the fouling cycle."""
    reward_fn = RewardFunction()
    _, fouling, = reward_fn(_telemetry(), dt_h=1.0, deposit_delta=0.2)
    _, cleaning = reward_fn(_telemetry(), dt_h=1.0, deposit_delta=-0.2)
    assert fouling.deposit > 0.0
    assert cleaning.deposit == 0.0


def test_the_breakdown_sums_to_the_reported_cost():
    reward_fn = RewardFunction()
    reward, breakdown = reward_fn(
        _telemetry(fuel_l=20.0, black_carbon_mg=5_000.0, genset_starts=1, curtailed_kw=10.0),
        dt_h=1.0,
        deposit_delta=0.05,
    )
    assert reward == pytest.approx(-breakdown.total_cost * reward_fn.weights.scale)
    assert breakdown.as_dict()["total_cost"] == pytest.approx(breakdown.total_cost)


def test_weights_are_configurable():
    cheap = RewardFunction(RewardWeights(fuel_per_l=1.0))
    dear = RewardFunction(RewardWeights(fuel_per_l=1000.0))
    telemetry = _telemetry(fuel_l=10.0)
    assert cheap(telemetry, 1.0, 0.0)[0] > dear(telemetry, 1.0, 0.0)[0]


# -- the environment agrees with the plant ------------------------------------


def test_disciplined_dispatch_scores_better_than_the_incumbent():
    """The reward must actually prefer the strategy the project advocates.

    A reward that ranks these two the wrong way round would train an agent
    towards the incumbent, however good the rest of the machinery is.
    """
    scores = {}
    for controller_cls in (LegacyNPlusOne, EfficientRuleBased):
        env = PolarMicrogridEnv(station="maitri", start=WINTER, periods=24 * 14, seed=2)
        env.reset(seed=2)
        controller = controller_cls(env.cfg)
        total = 0.0
        while True:
            command = controller.act(env.plant.observe(), env.plant)
            _, reward, terminated, truncated, _ = env.step(env.encode_command(command))
            total += reward
            if terminated or truncated:
                break
        scores[controller_cls.__name__] = total

    assert scores["EfficientRuleBased"] > scores["LegacyNPlusOne"]


def test_the_safety_layer_can_be_disabled_for_ablation():
    """Turning the guarantee off must be possible, and must visibly matter."""
    guarded = PolarMicrogridEnv(station="maitri", start=WINTER, periods=24 * 14, seed=3)
    unguarded = PolarMicrogridEnv(
        station="maitri", start=WINTER, periods=24 * 14, seed=3, apply_safety=False
    )
    summaries = []
    for env in (guarded, unguarded):
        env.reset(seed=3)
        env.action_space.seed(11)
        while True:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        summaries.append(env.summary())

    assert summaries[0]["critical_unserved_kwh"] == pytest.approx(0.0, abs=1e-9)
    assert summaries[1]["critical_unserved_kwh"] > 0.0, (
        "without the projection a random policy should endanger the station"
    )
