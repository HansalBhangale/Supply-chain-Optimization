# Stage 1 QUBO Package
"""
QUBO formulation and QAOA solver for strategic assignments.

Stage 1 solves binary assignment decisions:
- x_{i,j}: supplier i can supply warehouse j
- y_{j,k}: warehouse j serves customer k
"""

from .qubo_builder import QUBOBuilder
from .constraints import ConstraintEncoder
from .penalties import PenaltyCalculator
from .qaoa_solver import QAOASolver

__all__ = [
    "QUBOBuilder",
    "ConstraintEncoder", 
    "PenaltyCalculator",
    "QAOASolver",
]
