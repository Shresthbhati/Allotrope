"""A simple, explainable confidence/out-of-distribution signal for a learned controller.

Deliberately not a learned uncertainty model -- this project's own stated
preference is a simple, explainable baseline before anything more complex.
It combines three independently-checkable signals, each of which already
means something on its own without requiring a trained ensemble or a
calibrated probability:

  1. **Observation validity** -- `allotrope.safety.sensor_health`'s own
     CORRUPTED/STALE classification of the current reading. A controller
     acting on a reading already known to be implausible cannot be fully
     trusted regardless of what its policy network outputs.
  2. **Envelope distance** -- how far the current encoded observation sits
     from the mean of a *reference* set of observations, in per-feature
     standard deviations. This is explicitly not a claim about the RL
     agent's actual training data distribution: this module has no access
     to a training run's replay buffer. `EnvelopeStats.fit` takes whatever
     reference observations the caller supplies (e.g. a normal-operation
     evaluation trace); the honesty of "out of envelope" depends entirely
     on that reference set actually representing normal operation.
  3. **Safety intervention rate** -- how often, in a recent window, the
     safety projection has had to change what the controller proposed.
     This is a real, already-recorded signal (`SafetyReport`), not an
     invented one: a controller whose raw proposals keep getting
     overridden is not tracking the plant well, whatever the reason.

`ensemble_disagreement` is always reported as `None`, not faked as a
number: neither `BranchingDQN` nor `SDDPG` expose an ensemble or multiple
value-estimate heads through their public `.act()` interface (SDDPG's twin
critics exist internally for TD3-style training stability, not as a
surfaced uncertainty signal). A caller that later adds a real ensemble can
populate this field; nothing here pretends one already exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from allotrope.safety.projection import SafetyReport
from allotrope.safety.sensor_health import ObservationHealth, SensorStatus


@dataclass
class EnvelopeStats:
    """Per-feature mean/std of a reference set of (encoded) observations."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, observations: np.ndarray) -> "EnvelopeStats":
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 2 or observations.shape[0] < 2:
            raise ValueError("need at least 2 reference observations (rows) to fit a spread")
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        # A feature that is genuinely constant in the reference set (e.g. a
        # station-invariant scaling constant) would otherwise divide by
        # zero and manufacture an infinite z-score for any tiny numerical
        # noise; floor it instead of treating that as a real signal.
        std = np.where(std < 1e-6, 1e-6, std)
        return cls(mean=mean, std=std)

    def z_scores(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        return (observation - self.mean) / self.std

    def mean_squared_z(self, observation: np.ndarray) -> float:
        """The average squared z-score across features -- a diagonal
        Mahalanobis-style distance, and the metric this module actually
        uses for envelope distance.

        Deliberately not `max(|z|)`: a run against this project's own
        real observation encoding found that a single near-constant
        feature (a genset commitment flag that is `True` at almost every
        step, `False` only at an episode's very first step before
        anything has been committed) produces a z-score above 14 on that
        one legitimate transition purely because its reference standard
        deviation is small, while every other feature stays well within
        2 sigma. A max-based distance would call that single feature's
        rare-but-normal value out-of-envelope; averaging the squared
        z-scores lets 37 unremarkable features properly outweigh one
        naturally low-variance one.
        """
        z = self.z_scores(observation)
        return float(np.mean(z**2))


@dataclass
class ConfidenceAssessment:
    confidence: float
    """0 (no trust) to 1 (fully trusted). A product of the three component
    scores below, so any one bad signal can drag it down -- a controller
    is not trusted just because two of three checks look fine."""
    ood_detected: bool
    ood_reason: str | None
    ensemble_disagreement: float | None
    observation_status: SensorStatus
    envelope_distance: float | None
    """Mean squared z-score across encoded-observation features (see
    `EnvelopeStats.mean_squared_z`), or `None` if no envelope was
    configured. A well-calibrated reference set puts ordinary operation
    around 1.0; this project's own real observation encoding measures
    roughly 0.3-0.6 in steady operation (see `test_confidence.py`)."""
    recent_intervention_rate: float


@dataclass
class ControllerConfidence:
    """Stateful across a run: tracks a rolling window of recent safety
    interventions, so `intervention_rate` reflects recent behaviour rather
    than one step in isolation. One instance per episode/deployment run --
    a fresh instance should be used for a fresh run, or the intervention
    window will mix unrelated history in with it."""

    observation_health: ObservationHealth
    envelope: EnvelopeStats | None = None
    distance_threshold: float = 3.0
    """A mean-squared-z (see `EnvelopeStats.mean_squared_z`) above this is
    judged out-of-envelope. This project's own reference run measured
    ordinary operation at roughly 0.3-0.6, so 3.0 is a deliberately loose
    multiple of that -- meant to catch a state genuinely far outside
    anything the reference set saw, not to flag ordinary variation."""
    intervention_window: int = 24
    intervention_rate_threshold: float = 0.5
    _recent_interventions: deque = field(default_factory=deque, repr=False)

    def assess(
        self,
        raw_observation: dict,
        encoded_observation: np.ndarray | None = None,
        safety_report: SafetyReport | None = None,
    ) -> ConfidenceAssessment:
        obs_status = self.observation_health.worst(raw_observation)

        distance = None
        if self.envelope is not None and encoded_observation is not None:
            distance = self.envelope.mean_squared_z(encoded_observation)

        intervened = bool(safety_report and safety_report.interventions)
        self._recent_interventions.append(intervened)
        while len(self._recent_interventions) > self.intervention_window:
            self._recent_interventions.popleft()
        rate = (
            sum(self._recent_interventions) / len(self._recent_interventions)
            if self._recent_interventions
            else 0.0
        )

        reasons = []
        if obs_status != SensorStatus.VALID:
            reasons.append(f"observation_{obs_status.value}")
        if distance is not None and distance > self.distance_threshold:
            reasons.append(f"envelope_distance_{distance:.1f}")
        if rate > self.intervention_rate_threshold:
            reasons.append(f"safety_intervention_rate_{rate:.2f}")

        obs_component = 1.0 if obs_status == SensorStatus.VALID else 0.0
        envelope_component = (
            1.0
            if distance is None
            else float(np.clip(1.0 - distance / (2 * self.distance_threshold), 0.0, 1.0))
        )
        intervention_component = float(np.clip(1.0 - rate, 0.0, 1.0))
        confidence = obs_component * envelope_component * intervention_component

        return ConfidenceAssessment(
            confidence=confidence,
            ood_detected=bool(reasons),
            ood_reason=", ".join(reasons) if reasons else None,
            ensemble_disagreement=None,
            observation_status=obs_status,
            envelope_distance=distance,
            recent_intervention_rate=rate,
        )


__all__ = ["EnvelopeStats", "ConfidenceAssessment", "ControllerConfidence"]
