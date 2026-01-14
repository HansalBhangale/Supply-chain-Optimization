"""
Hybrid solver coordinator for two-stage optimization.

Implements Section 3: Two-Stage Stochastic Programming.
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
import time

from ..config import SupplyChainConfig
from ..data.models import SupplyChainNetwork
from ..data.scenarios import DemandScenario, generate_demand_scenarios
from ..data.distances import compute_distance_matrices, compute_lead_time_matrices
from ..stage1_qubo.qubo_builder import QUBOBuilder
from ..stage1_qubo.qaoa_solver import QAOASolver
from .repair import FeasibilityRepair
from ..stage2_milp.solver import MultiScenarioSolver


@dataclass
class HybridSolverResult:
    """Results from hybrid two-stage optimization."""
    
    # Stage 1 results
    x_assign: Dict[Tuple[int, int], int]  # Supplier-warehouse assignments
    y_assign: Dict[Tuple[int, int], int]  # Warehouse-customer assignments
    stage1_value: float  # QUBO objective value
    stage1_time: float   # Stage 1 solve time (seconds)
    
    # Feasibility repair stats
    repair_stats: Dict[str, int]
    feasibility_rate_before: float
    feasibility_rate_after: float
    
    # Stage 2 results
    expected_cost: float  # E_ω[Z_ω(x,y)]
    scenario_costs: Dict[int, float]  # Cost per scenario
    stage2_time: float  # Stage 2 solve time (seconds)
    
    # Total objective
    total_objective: float  # A(x,y) + E_ω[Z_ω]
    
    # Cost statistics
    cost_statistics: Dict[str, float]


class HybridSolver:
    """
    Two-stage hybrid quantum-classical solver.
    
    Stage 1: QAOA for strategic assignments (x, y)
    Stage 2: MILP for operational scheduling per scenario
    
    Total objective: Z_total(x,y) = A(x,y) + E_ω[Z_ω(x,y)]
    """
    
    def __init__(
        self,
        network: SupplyChainNetwork,
        config: SupplyChainConfig,
        scenarios: Optional[List[DemandScenario]] = None
    ):
        """
        Initialize hybrid solver.
        
        Args:
            network: Supply chain network
            config: Configuration parameters
            scenarios: Demand scenarios (generated if not provided)
        """
        self.network = network
        self.config = config
        
        # Ensure matrices are computed
        compute_distance_matrices(network)
        compute_lead_time_matrices(network)
        
        # Generate scenarios if not provided
        if scenarios is None:
            self.scenarios = generate_demand_scenarios(
                network,
                config.scenarios,
                random_seed=config.random_seed
            )
        else:
            self.scenarios = scenarios
        
        # Initialize components
        self.qubo_builder: Optional[QUBOBuilder] = None
        self.qaoa_solver: Optional[QAOASolver] = None
        self.repair: Optional[FeasibilityRepair] = None
        self.milp_solver: Optional[MultiScenarioSolver] = None
        
        # Results
        self.result: Optional[HybridSolverResult] = None
    
    def solve(
        self,
        use_classical_stage1: bool = False,
        milp_solver: Optional[str] = None,
        milp_time_limit: int = 300,
        warm_start: Optional[np.ndarray] = None
    ) -> HybridSolverResult:
        """
        Run full two-stage optimization.
        
        Args:
            use_classical_stage1: If True, use classical solver for Stage 1
            milp_solver: MILP solver ('PULP_CBC_CMD' or 'GUROBI_CMD')
            milp_time_limit: Time limit for Stage 2 MILP
            warm_start: Initial QAOA parameters for warm start
            
        Returns:
            HybridSolverResult with complete solution
        """
        print("=" * 60)
        print("HYBRID QUANTUM-CLASSICAL SUPPLY CHAIN OPTIMIZATION")
        print("=" * 60)
        
        # Stage 1: QAOA for strategic assignments
        print("\n[Stage 1] Solving QUBO with QAOA...")
        stage1_start = time.time()
        
        x_raw, y_raw, stage1_value = self._solve_stage1(
            use_classical=use_classical_stage1,
            warm_start=warm_start
        )
        
        stage1_time = time.time() - stage1_start
        print(f"  Stage 1 completed in {stage1_time:.2f}s")
        print(f"  QUBO objective: {stage1_value:.2f}")
        
        # Compute feasibility before repair
        self.repair = FeasibilityRepair(self.network, self.config)
        metrics_before = self.repair.compute_feasibility_metrics(x_raw, y_raw)
        feasibility_before = 1.0 - (metrics_before["total_violations"] / 
                                     max(1, self.network.num_customers))
        
        print(f"\n[Feasibility Repair]")
        print(f"  Violations before: {metrics_before['total_violations']}")
        
        # Apply feasibility repair
        x_assign, y_assign, repair_stats = self.repair.repair(x_raw, y_raw)
        
        metrics_after = self.repair.compute_feasibility_metrics(x_assign, y_assign)
        feasibility_after = 1.0 - (metrics_after["total_violations"] / 
                                    max(1, self.network.num_customers))
        
        print(f"  Repairs made: {sum(repair_stats.values())}")
        print(f"  Violations after: {metrics_after['total_violations']}")
        
        # Stage 2: MILP for operational scheduling
        print(f"\n[Stage 2] Solving MILP across {len(self.scenarios)} scenarios...")
        stage2_start = time.time()
        
        expected_cost, scenario_costs, cost_stats = self._solve_stage2(
            x_assign, y_assign,
            solver=milp_solver,
            time_limit=milp_time_limit
        )
        
        stage2_time = time.time() - stage2_start
        print(f"  Stage 2 completed in {stage2_time:.2f}s")
        print(f"  Expected operational cost: {expected_cost:.2f}")
        
        # Compute total objective
        total_objective = stage1_value + expected_cost
        
        print(f"\n[Results Summary]")
        print(f"  Total objective: {total_objective:.2f}")
        print(f"  - Stage 1 (assignment cost): {stage1_value:.2f}")
        print(f"  - Stage 2 (expected recourse): {expected_cost:.2f}")
        print(f"  Cost std dev: {cost_stats.get('std_cost', 0):.2f}")
        print(f"  CVaR (95%): {cost_stats.get('cvar_95', 0):.2f}")
        
        # Build result
        self.result = HybridSolverResult(
            x_assign=x_assign,
            y_assign=y_assign,
            stage1_value=stage1_value,
            stage1_time=stage1_time,
            repair_stats=repair_stats,
            feasibility_rate_before=feasibility_before,
            feasibility_rate_after=feasibility_after,
            expected_cost=expected_cost,
            scenario_costs=scenario_costs,
            stage2_time=stage2_time,
            total_objective=total_objective,
            cost_statistics=cost_stats
        )
        
        return self.result
    
    def _solve_stage1(
        self,
        use_classical: bool,
        warm_start: Optional[np.ndarray]
    ) -> Tuple[Dict, Dict, float]:
        """Solve Stage 1 QUBO."""
        # Build QUBO
        self.qubo_builder = QUBOBuilder(
            self.network,
            self.config,
            self.scenarios
        )
        qubo_matrix, constant, indexer = self.qubo_builder.build()
        
        print(f"  QUBO size: {qubo_matrix.shape[0]} variables")
        print(f"  - x variables: {indexer.num_x_variables}")
        print(f"  - y variables: {indexer.num_y_variables}")
        print(f"  - slack variables: {indexer.num_slack_variables}")
        
        # Solve
        self.qaoa_solver = QAOASolver(self.config.qaoa)
        
        if use_classical:
            bitstring, value = self.qaoa_solver.solve_classical(qubo_matrix, constant)
        else:
            bitstring, value = self.qaoa_solver.solve(
                qubo_matrix, constant, indexer, warm_start
            )
        
        # Decode solution
        x_assign, y_assign = self.qubo_builder.decode_solution(bitstring)
        
        return x_assign, y_assign, value
    
    def _solve_stage2(
        self,
        x_assign: Dict,
        y_assign: Dict,
        solver: Optional[str],
        time_limit: int
    ) -> Tuple[float, Dict[int, float], Dict[str, float]]:
        """Solve Stage 2 MILP across scenarios."""
        self.milp_solver = MultiScenarioSolver(
            self.network,
            self.config,
            self.scenarios
        )
        
        expected_cost = self.milp_solver.solve(
            x_assign, y_assign,
            solver=solver,
            time_limit=time_limit,
            parallel=True
        )
        
        scenario_costs = self.milp_solver.scenario_costs
        cost_stats = self.milp_solver.get_cost_statistics()
        
        return expected_cost, scenario_costs, cost_stats
    
    def get_network_summary(self) -> Dict:
        """Get summary of network assignments."""
        if self.result is None:
            return {}
        
        # Count active assignments
        active_sw = sum(1 for v in self.result.x_assign.values() if v == 1)
        active_wc = sum(1 for v in self.result.y_assign.values() if v == 1)
        
        # Warehouses serving customers
        active_warehouses = set()
        for (j, k), v in self.result.y_assign.items():
            if v == 1:
                active_warehouses.add(j)
        
        return {
            "active_supplier_warehouse_links": active_sw,
            "active_warehouse_customer_links": active_wc,
            "active_warehouses": len(active_warehouses),
            "total_suppliers": self.network.num_suppliers,
            "total_warehouses": self.network.num_warehouses,
            "total_customers": self.network.num_customers
        }
    
    def get_qaoa_parameters(self) -> Optional[np.ndarray]:
        """Get QAOA parameters for warm start in rolling horizon."""
        if self.qaoa_solver is not None:
            return self.qaoa_solver.get_qaoa_parameters()
        return None
