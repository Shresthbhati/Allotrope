"""An offline MILP optimization reference for the same problem the RL agents solve.

Answers a question this project could not answer before: how far is a
learned or rule-based controller from a strong optimization solution on the
identical scenario? `allotrope.optimization.milp.solve` is an offline oracle
-- it sees the whole horizon's weather/demand in advance, which no online
controller can -- so it is a ceiling to compare against, not a live
fallback and not a claim of global optimality (see `model.py`'s module
docstring for the physical simplifications this formulation makes to stay
linear).

Not wired into `allotrope.envs` or `allotrope.safety`, and not intended to
be: an offline, whole-horizon MILP cannot run inside a real-time control
loop. `allotrope.optimization.mpc` (future work) is the rolling-horizon,
online-appropriate use of the same model.
"""

from allotrope.optimization.milp import solve
from allotrope.optimization.model import MILPInstance, instance_from_plant
from allotrope.optimization.result import OptimizationResult

__all__ = [
    "MILPInstance",
    "instance_from_plant",
    "OptimizationResult",
    "solve",
]
