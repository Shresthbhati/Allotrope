"""Correctness tests for the confidence/OOD assessment.

Each of the three signals (observation validity, envelope distance, safety
intervention rate) is tested in isolation, then combined, against real
station data -- not synthetic numbers picked to make the module look good.
"""

from __future__ import annotations

import numpy as np
import pytest

from allotrope.config import load_station
from allotrope.control.baseline import EfficientRuleBased
from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.intelligence.confidence import ControllerConfidence, EnvelopeStats
from allotrope.safety.projection import Intervention, SafetyReport
from allotrope.safety.sensor_health import ObservationHealth, SensorStatus, specs_for_station
from allotrope.sim.runner import build_plant


@pytest.fixture(scope="module")
def cfg():
    return load_station("maitri")


@pytest.fixture(scope="module")
def normal_observation(cfg):
    plant = build_plant(cfg, start="2026-06-01", periods=24, seed=1)
    plant.reset()
    return plant.observe()


@pytest.fixture(scope="module")
def reference_envelope(cfg):
    plant = build_plant(cfg, start="2026-06-01", periods=200, seed=1)
    plant.reset()
    controller = EfficientRuleBased(cfg)
    codec_env = PolarMicrogridEnv(cfg, periods=2, apply_safety=False)
    encoded = []
    for _ in range(200):
        obs = plant.observe()
        codec_env.plant = plant
        encoded.append(codec_env._observe(obs=obs))
        plant.step(controller.act(obs, plant))
    return np.array(encoded)


def test_envelope_stats_needs_at_least_two_reference_rows():
    with pytest.raises(ValueError):
        EnvelopeStats.fit(np.zeros((1, 5)))


def test_envelope_stats_floors_a_zero_variance_feature_to_avoid_divide_by_zero():
    reference = np.ones((10, 3))
    stats = EnvelopeStats.fit(reference)
    # A feature identical in every reference row would otherwise produce
    # an infinite z-score for any nonzero deviation.
    z = stats.mean_squared_z(np.array([1.0, 1.0, 2.0]))
    assert np.isfinite(z)


def test_a_normal_observation_is_not_flagged_ood(cfg, normal_observation, reference_envelope):
    envelope = EnvelopeStats.fit(reference_envelope)
    controller_confidence = ControllerConfidence(
        observation_health=ObservationHealth(specs_for_station(cfg)), envelope=envelope
    )
    # A mid-episode row, not the reference set's first row: the very first
    # step of an episode is itself atypical (several genset commitment
    # flags sit at values they only ever take before anything has
    # started), which is a real, separate finding documented in
    # EnvelopeStats.mean_squared_z's docstring -- not what this test is
    # about.
    encoded = reference_envelope[100]
    result = controller_confidence.assess(normal_observation, encoded, None)
    assert not result.ood_detected
    assert result.confidence > 0.5
    assert result.observation_status == SensorStatus.VALID


def test_a_corrupted_observation_drives_confidence_to_zero(cfg, normal_observation):
    controller_confidence = ControllerConfidence(
        observation_health=ObservationHealth(specs_for_station(cfg))
    )
    corrupted = dict(normal_observation)
    corrupted["critical_load_kw"] = float("nan")
    result = controller_confidence.assess(corrupted, None, None)
    assert result.ood_detected
    assert result.confidence == 0.0
    assert "observation_corrupted" in result.ood_reason
    assert result.observation_status == SensorStatus.CORRUPTED


def test_an_observation_far_outside_the_reference_envelope_is_flagged(
    cfg, normal_observation, reference_envelope
):
    envelope = EnvelopeStats.fit(reference_envelope)
    controller_confidence = ControllerConfidence(
        observation_health=ObservationHealth(specs_for_station(cfg)), envelope=envelope
    )
    weird = reference_envelope[100].copy()
    weird[0] += 100.0
    result = controller_confidence.assess(normal_observation, weird, None)
    assert result.ood_detected
    assert "envelope_distance" in result.ood_reason
    assert result.envelope_distance > 3.0


def test_without_an_envelope_configured_only_observation_and_interventions_matter(
    cfg, normal_observation
):
    controller_confidence = ControllerConfidence(
        observation_health=ObservationHealth(specs_for_station(cfg))
    )
    result = controller_confidence.assess(normal_observation, np.array([1e9]), None)
    assert result.envelope_distance is None
    assert not result.ood_detected


def test_repeated_safety_interventions_raise_the_intervention_rate_and_flag_ood(
    cfg, normal_observation
):
    controller_confidence = ControllerConfidence(
        observation_health=ObservationHealth(specs_for_station(cfg)),
        intervention_window=10,
        intervention_rate_threshold=0.5,
    )
    result = None
    for i in range(10):
        report = SafetyReport()
        if i < 8:
            report.record(Intervention.SANITISED_NAN)
        result = controller_confidence.assess(normal_observation, None, report)
    assert result.recent_intervention_rate == pytest.approx(0.8)
    assert result.ood_detected
    assert "safety_intervention_rate" in result.ood_reason


def test_the_intervention_window_only_looks_at_recent_history(cfg, normal_observation):
    controller_confidence = ControllerConfidence(
        observation_health=ObservationHealth(specs_for_station(cfg)),
        intervention_window=5,
        intervention_rate_threshold=0.5,
    )
    for _ in range(5):
        report = SafetyReport()
        report.record(Intervention.SANITISED_NAN)
        controller_confidence.assess(normal_observation, None, report)
    # Five steps with no interventions should fully displace the earlier bad history.
    result = None
    for _ in range(5):
        result = controller_confidence.assess(normal_observation, None, SafetyReport())
    assert result.recent_intervention_rate == 0.0
    assert not result.ood_detected


def test_ensemble_disagreement_is_never_fabricated(cfg, normal_observation):
    """No ensemble exists in this codebase (see assessment.py's module
    docstring) -- this field must always be None, never a made-up number."""
    controller_confidence = ControllerConfidence(
        observation_health=ObservationHealth(specs_for_station(cfg))
    )
    result = controller_confidence.assess(normal_observation, None, None)
    assert result.ensemble_disagreement is None
