"""Scripted equipment failures for testing controller and safety-layer resilience.

`FaultInjector` wraps a `PolarMicrogridEnv` and enforces a fixed schedule of
`FaultEvent`s at the physical layer each failure belongs to (see
`injector.py`'s module docstring). Not wired into training or the safety
projection by default -- an opt-in tool for benchmarks, demos, and tests
that want to ask "what happens when something breaks," the same way
`allotrope.optimization` is an opt-in offline oracle.
"""

from allotrope.faults.events import FaultEvent, FaultKind
from allotrope.faults.injector import FaultInjector

__all__ = ["FaultEvent", "FaultKind", "FaultInjector"]
