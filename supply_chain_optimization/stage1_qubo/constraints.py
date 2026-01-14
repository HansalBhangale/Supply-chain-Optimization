"""
Constraint encoding for QUBO formulation.

Implements Section 4.2-4.3: Universal squared-equality penalty and encoded constraints.
"""

from typing import Dict, Tuple, List
import numpy as np

from .penalties import SlackVariableHelper


class ConstraintEncoder:
    """
    Encodes supply chain constraints as QUBO penalties.
    
    From Section 4.2 - Universal squared-equality penalty template:
    P = λ (T - Σ_r w_r z_r)²
    
    Expansion:
    P = λT² + Σ_r λ(w_r² - 2T·w_r) z_r + Σ_{r<s} (2λ w_r w_s) z_r z_s
    """
    
    def __init__(self, penalty_weights: Dict[str, float]):
        """
        Initialize constraint encoder.
        
        Args:
            penalty_weights: Dict mapping constraint name to λ value
        """
        self.penalty_weights = penalty_weights
        self.slack_helper = SlackVariableHelper()
    
    def encode_equality_constraint(
        self,
        qubo_matrix: np.ndarray,
        variable_indices: List[int],
        weights: List[float],
        target: float,
        penalty_weight: float
    ) -> float:
        """
        Encode equality constraint: Σ_r w_r z_r = T
        
        From Section 4.2:
        P = λ(T - Σ_r w_r z_r)²
        
        QUBO updates:
        Q[a,a] += λ(w_r² - 2T·w_r)
        Q[a,b] += 2λ·w_r·w_s (for r<s)
        const += λT²
        
        Args:
            qubo_matrix: QUBO matrix Q to update
            variable_indices: Indices of variables in the constraint
            weights: Coefficient w_r for each variable
            target: Target value T
            penalty_weight: Penalty λ
            
        Returns:
            Constant term added to QUBO
        """
        λ = penalty_weight
        T = target
        n = len(variable_indices)
        
        # Constant term: λT²
        constant = λ * T * T
        
        # Linear terms: Q[a,a] += λ(w_r² - 2T·w_r)
        for i in range(n):
            idx = variable_indices[i]
            w_r = weights[i]
            qubo_matrix[idx, idx] += λ * (w_r * w_r - 2 * T * w_r)
        
        # Quadratic terms: Q[a,b] += 2λ·w_r·w_s for r < s
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = variable_indices[i]
                idx_j = variable_indices[j]
                w_r, w_s = weights[i], weights[j]
                
                # Add to both Q[i,j] and Q[j,i] for symmetric matrix
                # Or just upper triangle: Q[i,j] += 2λ·w_r·w_s
                qubo_matrix[idx_i, idx_j] += 2 * λ * w_r * w_s
        
        return constant
    
    def encode_vendor_single_sourcing(
        self,
        qubo_matrix: np.ndarray,
        y_indices: Dict[Tuple[int, int], int],
        num_warehouses: int,
        customer_k: int
    ) -> float:
        """
        Encode: Σ_j y_{j,k} = 1 for vendor k (Section 4.3).
        
        Each vendor must be served by exactly one warehouse.
        
        Args:
            qubo_matrix: QUBO matrix to update
            y_indices: Dict mapping (j, k) to variable index
            num_warehouses: Number of warehouses |W|
            customer_k: Customer index k (must be vendor, k > 0)
            
        Returns:
            Constant term added
        """
        λ = self.penalty_weights.get("vendor_single_sourcing", 1000.0)
        
        # Get indices for all y_{j,k} for this customer
        indices = []
        for j in range(num_warehouses):
            if (j, customer_k) in y_indices:
                indices.append(y_indices[(j, customer_k)])
        
        # All weights are 1 (counting warehouses)
        weights = [1.0] * len(indices)
        target = 1.0  # Exactly one warehouse
        
        return self.encode_equality_constraint(
            qubo_matrix, indices, weights, target, λ
        )
    
    def encode_factory_redundancy(
        self,
        qubo_matrix: np.ndarray,
        y_indices: Dict[Tuple[int, int], int],
        num_warehouses: int,
        factory_k: int = 0,
        redundancy_count: int = 2
    ) -> float:
        """
        Encode: Σ_j y_{j,0} = 2 for factory (Section 4.3).
        
        Factory must be served by exactly 2 warehouses for redundancy.
        
        Args:
            qubo_matrix: QUBO matrix to update
            y_indices: Dict mapping (j, k) to variable index
            num_warehouses: Number of warehouses
            factory_k: Factory index (default 0)
            redundancy_count: Number of warehouses required
            
        Returns:
            Constant term added
        """
        λ = self.penalty_weights.get("factory_redundancy", 2000.0)
        
        indices = []
        for j in range(num_warehouses):
            if (j, factory_k) in y_indices:
                indices.append(y_indices[(j, factory_k)])
        
        weights = [1.0] * len(indices)
        target = float(redundancy_count)
        
        return self.encode_equality_constraint(
            qubo_matrix, indices, weights, target, λ
        )
    
    def encode_capacity_proxy(
        self,
        qubo_matrix: np.ndarray,
        y_indices: Dict[Tuple[int, int], int],
        slack_indices: List[int],
        warehouse_j: int,
        customer_demands: Dict[int, float],
        capacity_proxy: float
    ) -> float:
        """
        Encode capacity proxy with slack (Section 4.3):
        Σ_k (AvgDem_k · y_{j,k}) + Σ_b 2^b s_{j,b} = CapProxy_j
        
        Args:
            qubo_matrix: QUBO matrix to update
            y_indices: Dict mapping (j, k) to variable index
            slack_indices: Indices of slack variables s_{j,b}
            warehouse_j: Warehouse index j
            customer_demands: Dict mapping customer k to average demand
            capacity_proxy: Target capacity proxy value
            
        Returns:
            Constant term added
        """
        λ = self.penalty_weights.get("capacity_proxy", 500.0)
        
        indices = []
        weights = []
        
        # Add y_{j,k} variables with demand weights
        for k, demand in customer_demands.items():
            if (warehouse_j, k) in y_indices:
                indices.append(y_indices[(warehouse_j, k)])
                weights.append(demand)
        
        # Add slack variables with 2^b coefficients
        slack_coeffs = self.slack_helper.get_slack_coefficients(len(slack_indices))
        for i, slack_idx in enumerate(slack_indices):
            indices.append(slack_idx)
            weights.append(slack_coeffs[i])
        
        return self.encode_equality_constraint(
            qubo_matrix, indices, weights, capacity_proxy, λ
        )
    
    def encode_supplier_connectivity(
        self,
        qubo_matrix: np.ndarray,
        x_indices: Dict[Tuple[int, int], int],
        slack_indices: List[int],
        num_suppliers: int,
        warehouse_j: int,
        min_suppliers: int = 1
    ) -> float:
        """
        Encode supplier connectivity constraint with slack (Section 4.3):
        Σ_i x_{i,j} - min_suppliers - Σ_b 2^b r_{j,b} = 0
        
        Each active warehouse should have at least min_suppliers suppliers.
        
        Args:
            qubo_matrix: QUBO matrix to update
            x_indices: Dict mapping (i, j) to variable index
            slack_indices: Slack variable indices for inequality
            num_suppliers: Number of suppliers |S|
            warehouse_j: Warehouse index j
            min_suppliers: Minimum suppliers required
            
        Returns:
            Constant term added
        """
        λ = self.penalty_weights.get("supplier_connectivity", 800.0)
        
        indices = []
        weights = []
        
        # Add x_{i,j} variables with weight 1
        for i in range(num_suppliers):
            if (i, warehouse_j) in x_indices:
                indices.append(x_indices[(i, warehouse_j)])
                weights.append(1.0)
        
        # Add slack variables with negative 2^b coefficients (for ≥ constraint)
        slack_coeffs = self.slack_helper.get_slack_coefficients(len(slack_indices))
        for i, slack_idx in enumerate(slack_indices):
            indices.append(slack_idx)
            weights.append(-slack_coeffs[i])  # Negative for ≥ constraint
        
        target = float(min_suppliers)
        
        return self.encode_equality_constraint(
            qubo_matrix, indices, weights, target, λ
        )


def qubo_to_ising(qubo_matrix: np.ndarray, constant: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO to Ising formulation (Section 4.4).
    
    Map: z_a = (1 + σ_a)/2, σ_a ∈ {-1,+1}
    
    Given QUBO:
    E(z) = Σ_a Q_aa z_a + Σ_{a<b} Q_ab z_a z_b + const
    
    Ising coefficients:
    J_ab = Q_ab / 4
    h_a = (1/2)Q_aa + (1/4) Σ_{b≠a} Q_ab
    const' = const + (1/2) Σ_a Q_aa + (1/4) Σ_{a<b} Q_ab
    
    Args:
        qubo_matrix: QUBO Q matrix (n x n)
        constant: Constant offset in QUBO
        
    Returns:
        Tuple of (J coupling matrix, h local fields, new constant)
    """
    n = qubo_matrix.shape[0]
    
    # Make symmetric if not already
    Q = (qubo_matrix + qubo_matrix.T) / 2
    
    # J_ab = Q_ab / 4 (for a < b)
    J = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            J[a, b] = Q[a, b] / 4
            J[b, a] = J[a, b]  # Symmetric
    
    # h_a = (1/2)Q_aa + (1/4) Σ_{b≠a} Q_ab
    h = np.zeros(n)
    for a in range(n):
        h[a] = Q[a, a] / 2
        for b in range(n):
            if b != a:
                h[a] += Q[a, b] / 4
    
    # New constant
    new_const = constant
    new_const += np.sum(np.diag(Q)) / 2
    for a in range(n):
        for b in range(a + 1, n):
            new_const += Q[a, b] / 4
    
    return J, h, new_const
