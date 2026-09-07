"""A resilience benchmark: run controllers against scripted equipment
failures and report how each holds up, using the same evaluation entry
point (`allotrope.sim.runner.run_episode`) and the same fault semantics
(`allotrope.faults`) every other part of this project already uses.

Not a claim about which controller is "more resilient" in the abstract --
`run_benchmark` reports real numbers from real runs against the scenarios
in `scenarios.py`; what those numbers mean for a given comparison is for
the caller to interpret, not for this module to editorialise.
"""

from allotrope.resilience.benchmark import run_benchmark, run_scenario
from allotrope.resilience.scenarios import STANDARD_SCENARIOS

__all__ = ["run_benchmark", "run_scenario", "STANDARD_SCENARIOS"]
