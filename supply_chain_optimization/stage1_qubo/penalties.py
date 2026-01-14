"""
Penalty weight calculator implementing Section 5: Penalty Dominance Bounds.

Ensures constraint penalties dominate cost minimization in the QUBO.
"""

from typing import Tuple, Dict, List
import numpy as np


class PenaltyCalculator:
    """
    Computes penalty weights satisfying the dominance condition.
    
    From Section 5.1:
    If constraint m violated ⇒ P_m(z) ≥ 1, and λ_m > ΔC_max,
    then violating constraint m can never be beneficial.
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Initialize penalty calculator.
        
        Args:
            margin: Safety margin (default 10%) for penalty calculation
        """
        self.margin = margin
    
    def compute_delta_c_max(self, cost_coefficients: np.ndarray) -> float:
        """
        Compute ΔC_max for linear cost.
        
        From Section 5.2:
        If C(z) = Σ_a c_a z_a (linear), then:
        ΔC_max ≤ Σ_a |c_a|
        
        Args:
            cost_coefficients: Linear cost coefficients c_a
            
        Returns:
            Upper bound on maximum cost improvement
        """
        return float(np.sum(np.abs(cost_coefficients)))
    
    def compute_penalty_weight(
        self, 
        cost_coefficients: np.ndarray,
        constraint_priority: str = "hard"
    ) -> float:
        """
        Compute penalty weight for a constraint.
        
        From Section 5.2:
        λ_hard ≥ (1 + margin) · Σ_a |c_a|
        
        Args:
            cost_coefficients: Linear cost coefficients
            constraint_priority: "hard" or "soft"
            
        Returns:
            Penalty weight λ
        """
        delta_c_max = self.compute_delta_c_max(cost_coefficients)
        
        if constraint_priority == "hard":
            return (1 + self.margin) * delta_c_max
        else:
            # Soft constraints get smaller penalties
            return 0.5 * delta_c_max
    
    def integerize_weights(
        self, 
        weights: np.ndarray, 
        target: float,
        precision: int = 3
    ) -> Tuple[np.ndarray, float]:
        """
        Scale non-integer weights to integers (Section 5.3).
        
        If weights w_r are non-integers, scale by s to become integers:
        (T - Σ w_r z_r)² → (sT - Σ (s·w_r) z_r)²
        
        This preserves feasibility structure and ensures minimum violation ≥ 1.
        
        Args:
            weights: Original weights (potentially non-integer)
            target: Target value T in the constraint
            precision: Decimal places to preserve
            
        Returns:
            Tuple of (scaled integer weights, scaled target)
        """
        scale = 10 ** precision
        
        # Scale weights and target
        scaled_weights = np.round(weights * scale).astype(np.int64)
        scaled_target = round(target * scale)
        
        return scaled_weights, float(scaled_target)
    
    def compute_all_penalties(
        self,
        cost_coefficients: np.ndarray,
        num_constraints: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Compute penalty weights for all constraint types.
        
        Args:
            cost_coefficients: Linear cost coefficients
            num_constraints: Dict with counts of each constraint type
            
        Returns:
            Dict mapping constraint name to penalty weight
        """
        base_penalty = self.compute_penalty_weight(cost_coefficients)
        
        # Scale penalties by number of constraints to maintain dominance
        penalties = {}
        
        # Hard constraints (must be satisfied)
        penalties["vendor_single_sourcing"] = base_penalty * 2.0
        penalties["factory_redundancy"] = base_penalty * 3.0  # Factory is critical
        
        # Medium constraints
        penalties["supplier_connectivity"] = base_penalty * 1.5
        
        # Soft constraints (capacity proxy)
        penalties["capacity_proxy"] = base_penalty * 0.8
        
        return penalties


class SlackVariableHelper:
    """
    Helper for encoding slack variables in QUBO constraints.
    
    From Section 4.3:
    Inequality constraints use binary slack variables:
    Σk (AvgDem_k · y_{j,k}) + Σb 2^b s_{j,b} = CapProxy_j
    """
    
    @staticmethod
    def compute_num_slack_bits(max_slack_value: float) -> int:
        """
        Compute number of bits needed for slack variable.
        
        Args:
            max_slack_value: Maximum value the slack can take
            
        Returns:
            Number of binary slack bits needed
        """
        if max_slack_value <= 0:
            return 0
        return int(np.ceil(np.log2(max_slack_value + 1)))
    
    @staticmethod
    def get_slack_coefficients(num_bits: int) -> np.ndarray:
        """
        Get coefficients for slack variable encoding.
        
        Returns [2^0, 2^1, 2^2, ...] = [1, 2, 4, ...]
        
        Args:
            num_bits: Number of slack bits
            
        Returns:
            Array of powers of 2
        """
        return np.array([2**b for b in range(num_bits)], dtype=np.float64)
    
    @staticmethod
    def decode_slack_value(slack_bits: np.ndarray) -> int:
        """
        Decode slack value from binary representation.
        
        Args:
            slack_bits: Array of binary slack bits [s_0, s_1, ...]
            
        Returns:
            Integer slack value
        """
        coeffs = SlackVariableHelper.get_slack_coefficients(len(slack_bits))
        return int(np.dot(coeffs, slack_bits))
