"""
QUBO matrix builder for Stage 1 strategic assignment optimization.

Implements Section 4: Stage 1 QUBO Derivations.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np

from ..config import SupplyChainConfig
from ..data.models import SupplyChainNetwork
from ..data.scenarios import DemandScenario, compute_average_demand
from .penalties import PenaltyCalculator, SlackVariableHelper
from .constraints import ConstraintEncoder


class VariableIndexer:
    """
    Maps binary decision variables to QUBO matrix indices.
    
    Variables:
        x_{i,j} ∈ {0,1}: supplier i can supply warehouse j
        y_{j,k} ∈ {0,1}: warehouse j serves customer k
        s_{j,b}: slack bits for capacity proxy
        r_{j,b}: slack bits for supplier connectivity
    """
    
    def __init__(
        self,
        num_suppliers: int,
        num_warehouses: int,
        num_customers: int,
        num_slack_bits_capacity: int = 4,
        num_slack_bits_connectivity: int = 3
    ):
        self.num_s = num_suppliers
        self.num_w = num_warehouses
        self.num_c = num_customers
        self.slack_bits_cap = num_slack_bits_capacity
        self.slack_bits_conn = num_slack_bits_connectivity
        
        # Variable indices
        self.x_indices: Dict[Tuple[int, int], int] = {}  # (i, j) -> index
        self.y_indices: Dict[Tuple[int, int], int] = {}  # (j, k) -> index
        self.slack_cap_indices: Dict[int, List[int]] = {}  # j -> [indices]
        self.slack_conn_indices: Dict[int, List[int]] = {}  # j -> [indices]
        
        self._build_indices()
    
    def _build_indices(self):
        """Build variable index mappings."""
        idx = 0
        
        # x_{i,j}: supplier-warehouse assignments
        for i in range(self.num_s):
            for j in range(self.num_w):
                self.x_indices[(i, j)] = idx
                idx += 1
        
        # y_{j,k}: warehouse-customer assignments
        for j in range(self.num_w):
            for k in range(self.num_c):
                self.y_indices[(j, k)] = idx
                idx += 1
        
        # Slack variables for capacity proxy (per warehouse)
        for j in range(self.num_w):
            self.slack_cap_indices[j] = []
            for b in range(self.slack_bits_cap):
                self.slack_cap_indices[j].append(idx)
                idx += 1
        
        # Slack variables for supplier connectivity (per warehouse)
        for j in range(self.num_w):
            self.slack_conn_indices[j] = []
            for b in range(self.slack_bits_conn):
                self.slack_conn_indices[j].append(idx)
                idx += 1
        
        self.total_variables = idx
    
    @property
    def num_x_variables(self) -> int:
        """Number of x_{i,j} variables."""
        return self.num_s * self.num_w
    
    @property
    def num_y_variables(self) -> int:
        """Number of y_{j,k} variables."""
        return self.num_w * self.num_c
    
    @property
    def num_slack_variables(self) -> int:
        """Total number of slack variables."""
        return self.num_w * (self.slack_bits_cap + self.slack_bits_conn)


class QUBOBuilder:
    """
    Builds the QUBO matrix for Stage 1 optimization.
    
    QUBO energy form (Section 4.1):
    E(z) = Σ_a Q_aa z_a + Σ_{a<b} Q_ab z_a z_b + const
    
    Components:
    - Assignment cost terms (linear in x, y)
    - Constraint penalties (quadratic)
    """
    
    def __init__(
        self,
        network: SupplyChainNetwork,
        config: SupplyChainConfig,
        scenarios: Optional[List[DemandScenario]] = None
    ):
        """
        Initialize QUBO builder.
        
        Args:
            network: Supply chain network structure
            config: Configuration parameters
            scenarios: Demand scenarios for computing average demand
        """
        self.network = network
        self.config = config
        self.scenarios = scenarios
        
        # Initialize indexer
        self.indexer = VariableIndexer(
            num_suppliers=network.num_suppliers,
            num_warehouses=network.num_warehouses,
            num_customers=network.num_customers
        )
        
        # Initialize QUBO matrix
        n = self.indexer.total_variables
        self.qubo_matrix = np.zeros((n, n), dtype=np.float64)
        self.constant = 0.0
        
        # Initialize helpers
        self.penalty_calc = PenaltyCalculator(margin=config.penalties.margin)
        self.constraint_encoder: Optional[ConstraintEncoder] = None
    
    def build(self) -> Tuple[np.ndarray, float, VariableIndexer]:
        """
        Build the complete QUBO matrix.
        
        Returns:
            Tuple of (QUBO matrix, constant offset, variable indexer)
        """
        # 1. Compute cost coefficients
        cost_coeffs = self._compute_cost_coefficients()
        
        # 2. Add cost terms to QUBO diagonal
        self._add_cost_terms(cost_coeffs)
        
        # 3. Compute penalty weights
        penalty_weights = self.penalty_calc.compute_all_penalties(
            cost_coeffs,
            num_constraints={
                "vendor_single_sourcing": self.network.num_vendors,
                "factory_redundancy": 1,
                "capacity_proxy": self.network.num_warehouses,
                "supplier_connectivity": self.network.num_warehouses
            }
        )
        
        # Override with user-specified penalties if provided
        if self.config.penalties.vendor_single_sourcing > 0:
            penalty_weights["vendor_single_sourcing"] = self.config.penalties.vendor_single_sourcing
        if self.config.penalties.factory_redundancy > 0:
            penalty_weights["factory_redundancy"] = self.config.penalties.factory_redundancy
        
        # 4. Initialize constraint encoder
        self.constraint_encoder = ConstraintEncoder(penalty_weights)
        
        # 5. Add constraint penalties
        self._add_vendor_single_sourcing_constraints()
        self._add_factory_redundancy_constraint()
        self._add_capacity_proxy_constraints()
        self._add_supplier_connectivity_constraints()
        
        return self.qubo_matrix, self.constant, self.indexer
    
    def _compute_cost_coefficients(self) -> np.ndarray:
        """
        Compute linear cost coefficients for assignment variables.
        
        Cost A(x,y) approximation: sum of distance-weighted assignments.
        """
        n = self.indexer.total_variables
        costs = np.zeros(n)
        
        c_var = self.config.costs.variable_cost_per_km
        
        # Cost for x_{i,j}: variable_cost × distance
        for (i, j), idx in self.indexer.x_indices.items():
            if self.network.distance_sw is not None:
                costs[idx] = c_var * self.network.distance_sw[i, j]
            else:
                costs[idx] = c_var * 50.0  # Default distance estimate
        
        # Cost for y_{j,k}: variable_cost × distance × avg_demand factor
        for (j, k), idx in self.indexer.y_indices.items():
            if self.network.distance_wc is not None:
                dist = self.network.distance_wc[j, k]
            else:
                dist = 30.0  # Default distance estimate
            
            # Weight by customer importance
            customer = self.network.get_customer(k)
            demand_factor = customer.average_demand / 50.0  # Normalized
            costs[idx] = c_var * dist * demand_factor
        
        return costs
    
    def _add_cost_terms(self, cost_coefficients: np.ndarray):
        """Add linear cost terms to QUBO diagonal."""
        np.fill_diagonal(self.qubo_matrix, cost_coefficients)
    
    def _add_vendor_single_sourcing_constraints(self):
        """Add constraints: Σ_j y_{j,k} = 1 for each vendor k."""
        for k in range(1, self.network.num_customers):  # Skip factory (k=0)
            self.constant += self.constraint_encoder.encode_vendor_single_sourcing(
                self.qubo_matrix,
                self.indexer.y_indices,
                self.network.num_warehouses,
                customer_k=k
            )
    
    def _add_factory_redundancy_constraint(self):
        """Add constraint: Σ_j y_{j,0} = 2 for factory."""
        self.constant += self.constraint_encoder.encode_factory_redundancy(
            self.qubo_matrix,
            self.indexer.y_indices,
            self.network.num_warehouses,
            factory_k=0,
            redundancy_count=self.config.penalties.factory_redundancy_count
        )
    
    def _add_capacity_proxy_constraints(self):
        """Add capacity proxy constraints for each warehouse."""
        # Compute average customer demands
        customer_demands = {
            k: c.average_demand 
            for k, c in enumerate(self.network.customers)
        }
        
        for j in range(self.network.num_warehouses):
            warehouse = self.network.get_warehouse(j)
            # Capacity proxy = fraction of warehouse capacity
            capacity_proxy = warehouse.capacity * 0.8
            
            self.constant += self.constraint_encoder.encode_capacity_proxy(
                self.qubo_matrix,
                self.indexer.y_indices,
                self.indexer.slack_cap_indices[j],
                warehouse_j=j,
                customer_demands=customer_demands,
                capacity_proxy=capacity_proxy
            )
    
    def _add_supplier_connectivity_constraints(self):
        """Add supplier connectivity constraints for each warehouse."""
        for j in range(self.network.num_warehouses):
            self.constant += self.constraint_encoder.encode_supplier_connectivity(
                self.qubo_matrix,
                self.indexer.x_indices,
                self.indexer.slack_conn_indices[j],
                self.network.num_suppliers,
                warehouse_j=j,
                min_suppliers=1
            )
    
    def decode_solution(
        self,
        bitstring: np.ndarray
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        """
        Decode QUBO solution bitstring to assignment dictionaries.
        
        Args:
            bitstring: Binary solution array
            
        Returns:
            Tuple of (x_assign dict, y_assign dict)
        """
        x_assign = {}
        for (i, j), idx in self.indexer.x_indices.items():
            x_assign[(i, j)] = int(bitstring[idx])
        
        y_assign = {}
        for (j, k), idx in self.indexer.y_indices.items():
            y_assign[(j, k)] = int(bitstring[idx])
        
        return x_assign, y_assign
    
    def compute_energy(self, bitstring: np.ndarray) -> float:
        """
        Compute QUBO energy for a given bitstring.
        
        E(z) = z^T Q z + const
        
        Args:
            bitstring: Binary solution array
            
        Returns:
            Energy value
        """
        return float(bitstring @ self.qubo_matrix @ bitstring + self.constant)
