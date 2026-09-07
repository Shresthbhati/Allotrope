"""The learned agents: unit-level correctness, then the same safety guarantee
`tests/test_safety.py` demands of every other policy.

An untrained, effectively random network is exactly the adversarial policy
class `test_safety.py` already attacks: no gradient step is required for a
`BranchingDQN`/`SDDPG` pair to propose the same kind of nonsense a fuzzer
would. So the pipeline this file cares about most is

    HybridAgent (raw proposal) -> GuardedController -> plant

under the same random, seeded, freshly-initialised conditions, with
Hypothesis driving the seeds and the initial plant state -- 25 examples per
property below, not a fixed handful chosen by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from allotrope.agents.dqn import BranchingDQN, DQNConfig
from allotrope.agents.hybrid import HybridAgent
from allotrope.agents.replay_buffer import ReplayBuffer, Transition
from allotrope.agents.sddpg import SDDPG, SDDPGConfig
from allotrope.config import load_station
from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.safety.fallback import GuardedController
from allotrope.sim.runner import build_plant, run_episode

WINTER = "2026-06-01"


def _dims(cfg):
    # Mirrors PolarMicrogridEnv._observation_width exactly rather than
    # importing it, so a test-only PolarMicrogridEnv doesn't have to be
    # constructed just to learn a dimension -- but that means this must be
    # kept in sync by hand; a mismatch fails loudly (a matmul shape error),
    # never silently, as it did the one time this actually drifted.
    obs_dim = 14 + 6 * len(cfg.gensets) + 3 * len(cfg.storage)
    n_g = len(cfg.gensets)
    dispatch_dim = n_g + len(cfg.storage) + 1
    return obs_dim, n_g, dispatch_dim


# -- networks and buffer -------------------------------------------------------


def test_branching_q_network_shape():
    cfg = load_station("maitri")
    obs_dim, n_g, _ = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=0))
    action = dqn.act(np.zeros(obs_dim, dtype=np.float32))
    assert action.shape == (n_g,)
    assert set(np.unique(action)).issubset({0.0, 1.0})


def test_dqn_epsilon_decays_and_is_bounded():
    cfg = load_station("maitri")
    obs_dim, n_g, _ = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_steps=10))
    assert dqn.epsilon() == pytest.approx(1.0)
    for _ in range(50):
        dqn.act(np.zeros(obs_dim, dtype=np.float32))
    assert dqn.epsilon() == pytest.approx(0.05)


def test_dqn_deterministic_action_is_reproducible():
    cfg = load_station("maitri")
    obs_dim, n_g, _ = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=0))
    obs = np.random.default_rng(0).uniform(-1, 1, obs_dim).astype(np.float32)
    a1 = dqn.act(obs, deterministic=True)
    a2 = dqn.act(obs, deterministic=True)
    np.testing.assert_array_equal(a1, a2)


def test_sddpg_action_is_bounded_to_unit_range():
    cfg = load_station("maitri")
    obs_dim, _, dispatch_dim = _dims(cfg)
    agent = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=0))
    rng = np.random.default_rng(1)
    for _ in range(20):
        obs = rng.uniform(-5, 5, obs_dim).astype(np.float32)
        action = agent.act(obs, deterministic=False)
        assert action.shape == (dispatch_dim,)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_sddpg_deterministic_action_is_reproducible():
    cfg = load_station("maitri")
    obs_dim, _, dispatch_dim = _dims(cfg)
    agent = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=0))
    obs = np.random.default_rng(0).uniform(-1, 1, obs_dim).astype(np.float32)
    a1 = agent.act(obs, deterministic=True)
    a2 = agent.act(obs, deterministic=True)
    np.testing.assert_array_almost_equal(a1, a2)


def test_replay_buffer_wraps_and_samples_added_shape():
    buf = ReplayBuffer(capacity=8, obs_dim=4, n_gensets=2, dispatch_dim=3)
    rng = np.random.default_rng(0)
    for i in range(20):
        buf.add(
            Transition(
                obs=np.full(4, i, dtype=np.float32),
                genset_on=np.array([1.0, 0.0], dtype=np.float32),
                dispatch=np.zeros(3, dtype=np.float32),
                reward=float(i),
                next_obs=np.full(4, i + 1, dtype=np.float32),
                done=False,
            )
        )
    assert len(buf) == 8  # capped, never exceeds capacity
    batch = buf.sample(4, rng)
    assert batch["obs"].shape == (4, 4)
    assert batch["genset_on"].shape == (4, 2)
    assert batch["dispatch"].shape == (4, 3)


# -- learning actually moves the networks --------------------------------------


def test_dqn_update_reduces_loss_on_a_fixed_batch():
    cfg = load_station("maitri")
    obs_dim, n_g, _ = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=0, lr=1e-2))
    rng = np.random.default_rng(0)
    batch = {
        "obs": rng.uniform(-1, 1, (64, obs_dim)).astype(np.float32),
        "next_obs": rng.uniform(-1, 1, (64, obs_dim)).astype(np.float32),
        "genset_on": rng.integers(0, 2, (64, n_g)).astype(np.float32),
        "reward": rng.uniform(-1, 1, 64).astype(np.float32),
        "done": np.zeros(64, dtype=np.float32),
    }
    losses = [dqn.update(batch)["dqn_loss"] for _ in range(30)]
    assert losses[-1] < losses[0]


def test_sddpg_update_runs_without_nan_and_updates_critic():
    cfg = load_station("maitri")
    obs_dim, _, dispatch_dim = _dims(cfg)
    agent = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=0))
    rng = np.random.default_rng(0)
    batch = {
        "obs": rng.uniform(-1, 1, (64, obs_dim)).astype(np.float32),
        "next_obs": rng.uniform(-1, 1, (64, obs_dim)).astype(np.float32),
        "dispatch": rng.uniform(-1, 1, (64, dispatch_dim)).astype(np.float32),
        "reward": rng.uniform(-1, 1, 64).astype(np.float32),
        "done": np.zeros(64, dtype=np.float32),
    }
    before = [p.clone() for p in agent.critic.parameters()]
    metrics = agent.update(batch)
    assert np.isfinite(metrics["sddpg_critic_loss"])
    after = list(agent.critic.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_checkpoint_round_trip_preserves_behaviour(tmp_path):
    cfg = load_station("maitri")
    obs_dim, n_g, dispatch_dim = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=0))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=0))
    obs = np.random.default_rng(2).uniform(-1, 1, obs_dim).astype(np.float32)
    before_dqn = dqn.act(obs, deterministic=True)
    before_sddpg = sddpg.act(obs, deterministic=True)

    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "obs_dim": obs_dim, "n_gensets": n_g, "dispatch_dim": dispatch_dim,
            "dqn": dqn.state_dict(), "sddpg": sddpg.state_dict(),
        },
        path,
    )
    state = torch.load(path, weights_only=True)
    dqn2 = BranchingDQN(state["obs_dim"], state["n_gensets"], DQNConfig(seed=99))
    sddpg2 = SDDPG(state["obs_dim"], state["dispatch_dim"], SDDPGConfig(seed=99))
    dqn2.load_state_dict(state["dqn"])
    sddpg2.load_state_dict(state["sddpg"])

    np.testing.assert_array_equal(before_dqn, dqn2.act(obs, deterministic=True))
    np.testing.assert_array_almost_equal(before_sddpg, sddpg2.act(obs, deterministic=True))


# -- training loop integration: the environment, safely, end to end -----------


def test_a_few_training_steps_run_without_nan_or_crashing():
    cfg = load_station("maitri")
    obs_dim, n_g, dispatch_dim = _dims(cfg)
    env = PolarMicrogridEnv(cfg, periods=48, episode_steps=24, apply_safety=True, seed=0)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=0))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=0))
    buffer = ReplayBuffer(200, obs_dim, n_g, dispatch_dim)
    rng = np.random.default_rng(0)

    obs, _ = env.reset(seed=0)
    for _ in range(60):
        genset_on = dqn.act(obs, deterministic=False)
        dispatch = sddpg.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(
            {"genset_on": genset_on.astype(np.int8), "dispatch": dispatch}
        )
        assert np.isfinite(reward)
        assert np.all(np.isfinite(next_obs))
        buffer.add(Transition(obs, genset_on, dispatch, float(reward), next_obs, terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
        if len(buffer) >= 32:
            batch = buffer.sample(32, rng)
            m1 = dqn.update(batch)
            m2 = sddpg.update(batch)
            assert all(np.isfinite(v) for v in m1.values())
            assert all(np.isfinite(v) for v in m2.values())


def test_checkpoint_every_saves_progress_before_training_finishes(tmp_path):
    """A run killed partway through must not lose everything.

    Found the hard way: allotrope.train only wrote checkpoint.pt after the
    full run completed, so a training process interrupted by anything
    outside this codebase (a killed machine, not a code failure) lost every
    step of progress with nothing to resume from. --checkpoint-every writes
    the same file at a fixed cadence instead of only at the end.
    """
    from allotrope.train import train

    saved_at_steps: list[int] = []
    real_save = torch.save

    def counting_save(obj, path):
        saved_at_steps.append(obj["dqn"]["env_steps"])
        real_save(obj, path)

    import unittest.mock as mock

    with mock.patch("allotrope.train.torch.save", side_effect=counting_save):
        train(
            agent_kind="hybrid",
            station="maitri",
            total_steps=20,
            seed=0,
            episode_steps=24,
            warmup_steps=1,
            buffer_capacity=50,
            runs_dir=tmp_path,
            checkpoint_every=5,
        )

    # Steps 5, 10, 15 from the periodic branch, plus the unconditional final
    # save at step 20 -- four writes, not one, and each one strictly later
    # than the last (env_steps is monotonic), so an interrupted run always
    # has a checkpoint reflecting real, recent progress on disk.
    assert saved_at_steps == sorted(saved_at_steps)
    assert len(saved_at_steps) == 4
    assert saved_at_steps[-1] == 20


# -- the safety guarantee, for an untrained hybrid agent specifically ---------


def _assert_station_safe(summary, label=""):
    assert summary["critical_unserved_kwh"] == pytest.approx(0.0, abs=1e-9), (
        f"{label}: life support went unserved"
    )
    assert summary["freeze_violation_steps"] == 0.0, f"{label}: the station froze"


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_an_untrained_hybrid_agent_cannot_endanger_the_station(seed):
    """An untrained network is, functionally, an adversarial policy.

    Hypothesis drives the seed rather than a fixed sample: both the network
    initialisation and the plant's weather/demand realisation vary, and the
    guarantee must hold for all of it, not just the handful of seeds a human
    would have picked by hand.
    """
    cfg = load_station("maitri")
    obs_dim, n_g, dispatch_dim = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=seed))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=seed))
    hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=False)
    guard = GuardedController(cfg, agent=hybrid)

    plant = build_plant(cfg, start=WINTER, periods=24 * 5, seed=seed)
    result = run_episode(plant, guard)
    _assert_station_safe(result.summary, f"untrained hybrid seed={seed}")


def test_guarded_hybrid_agent_survives_a_corrupted_checkpoint():
    """The guard must still hold when the agent's weights are actively broken,
    not merely untrained -- the failure mode a corrupted model file produces."""
    cfg = load_station("maitri")
    obs_dim, n_g, dispatch_dim = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=0))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=0))
    with torch.no_grad():
        for p in list(dqn.online.parameters()) + list(sddpg.actor.parameters()):
            p.fill_(float("nan"))

    hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=True)
    guard = GuardedController(cfg, agent=hybrid)
    plant = build_plant(cfg, start=WINTER, periods=24 * 5, seed=1)
    result = run_episode(plant, guard)
    _assert_station_safe(result.summary, "NaN-corrupted hybrid agent")


def test_offline_evaluation_is_reproducible_across_runs():
    """A deterministic checkpoint evaluated offline must give the same answer
    every time -- not just the same action for the same observation.

    This guards against a real bug found in this project's own adversarial
    audit: `GuardedController`'s real-time latency budget (a genuinely
    correct safety property for actual control) silently substitutes the
    deterministic fallback whenever a forward pass exceeds it, and because
    that measurement is a wall-clock one, it made offline evaluation
    non-reproducible -- the same checkpoint and seed produced different
    genset-starts counts (489-527 observed across six runs) purely from
    CPU scheduling jitter. `enforce_latency_budget=False` is what
    `allotrope.evaluate`, `allotrope.evaluate_scenarios`, and
    `allotrope.federated.round`'s validator all now set for exactly this
    reason; this test locks in that repeated full-episode runs, not just
    repeated single-action calls, agree exactly.
    """
    cfg = load_station("maitri")
    obs_dim, n_g, dispatch_dim = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=0))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=0))
    hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=True)

    summaries = []
    for _ in range(3):
        guard = GuardedController(cfg, agent=hybrid, enforce_latency_budget=False)
        plant = build_plant(cfg, periods=24 * 20, seed=1)
        summaries.append(run_episode(plant, guard).summary)

    reference = summaries[0]
    for i, summary in enumerate(summaries[1:], start=1):
        assert summary == reference, f"run {i} disagreed with run 0: {summary} vs {reference}"


def test_hybrid_agent_matches_the_action_space_contract():
    """decode_action must never be handed something outside [-1, 1] or the wrong shape."""
    cfg = load_station("maitri")
    obs_dim, n_g, dispatch_dim = _dims(cfg)
    dqn = BranchingDQN(obs_dim, n_g, DQNConfig(seed=3))
    sddpg = SDDPG(obs_dim, dispatch_dim, SDDPGConfig(seed=3))
    hybrid = HybridAgent(cfg, dqn, sddpg, deterministic=False)

    plant = build_plant(cfg, start=WINTER, periods=48, seed=3)
    plant.reset()
    for _ in range(20):
        command = hybrid.act(plant.observe(), plant)
        assert len(command.genset_on) == n_g
        assert len(command.genset_setpoint_kw) == n_g
        assert len(command.battery_kw) == len(cfg.storage)
        assert np.isfinite(command.snow_melt_kw)
        plant.step(command)
