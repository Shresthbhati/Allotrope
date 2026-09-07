"""A simple, explainable confidence/OOD signal for a learned controller.

See `assessment.py`'s module docstring for the three signals combined and,
just as importantly, what is deliberately *not* faked (`ensemble_disagreement`
is always `None` -- no ensemble exists in this codebase yet).
"""

from allotrope.intelligence.confidence.assessment import (
    ConfidenceAssessment,
    ControllerConfidence,
    EnvelopeStats,
)

__all__ = ["ConfidenceAssessment", "ControllerConfidence", "EnvelopeStats"]
