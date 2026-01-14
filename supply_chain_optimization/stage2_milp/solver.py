"""
Multi-scenario solver for Stage 2 MILP.

Implements the expected recourse cost calculation across scenarios.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import SupplyChainConfig
from ..data.models import SupplyChainNetwork
from ..data.scenarios import DemandScenario
from .milp_model import Stage2MILP


class MultiScenarioSolver:
    """
    Solves Stage 2 MILP across multiple demand scenarios.
    
    Computes E_ω[Z_ω(x,y)] = Σ_ω π_ω · Z_ω(x,y)
    """
    
    def __init__(
        self,
        network: SupplyChainNetwork,
        config: SupplyChainConfig,
        scenarios: List[DemandScenario]
    ):
        """
        Initialize multi-scenario solver.
        
        Args:
            network: Supply chain network
            config: Configuration parameters
            scenarios: List of demand scenarios with probabilities
        """
        self.network = network
        self.config = config
        self.scenarios = scenarios
        
        # Results storage
        self.scenario_costs: Dict[int, float] = {}
        self.scenario_solutions: Dict[int, Dict] = {}
        self.expected_cost: Optional[float] = None
    
    def solve(
        self,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int],
        solver: Optional[str] = None,
        time_limit: int = 300,
        parallel: bool = True
    ) -> float:
        """
        Solve Stage 2 across all scenarios.
        
        Args:
            x_assign: Stage 1 supplier-warehouse assignments
            y_assign: Stage 1 warehouse-customer assignments
            solver: MILP solver to use
            time_limit: Time limit per scenario
            parallel: If True, solve scenarios in parallel
            
        Returns:
            Expected recourse cost E_ω[Z_ω]
        """
        if parallel and len(self.scenarios) > 1:
            return self._solve_parallel(x_assign, y_assign, solver, time_limit)
        else:
            return self._solve_sequential(x_assign, y_assign, solver, time_limit)
    
    def _solve_sequential(
        self,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int],
        solver: Optional[str],
        time_limit: int
    ) -> float:
        """Solve scenarios sequentially."""
        expected_cost = 0.0
        
        for scenario in self.scenarios:
            cost = self._solve_single_scenario(
                x_assign, y_assign, scenario, solver, time_limit
            )
            self.scenario_costs[scenario.id] = cost
            expected_cost += scenario.probability * cost
        
        self.expected_cost = expected_cost
        return expected_cost
    
    def _solve_parallel(
        self,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int],
        solver: Optional[str],
        time_limit: int
    ) -> float:
        """Solve scenarios in parallel using ThreadPoolExecutor."""
        expected_cost = 0.0
        
        with ThreadPoolExecutor(max_workers=min(4, len(self.scenarios))) as executor:
            futures = {
                executor.submit(
                    self._solve_single_scenario,
                    x_assign, y_assign, scenario, solver, time_limit
                ): scenario
                for scenario in self.scenarios
            }
            
            for future in as_completed(futures):
                scenario = futures[future]
                try:
                    cost = future.result()
                    self.scenario_costs[scenario.id] = cost
                    expected_cost += scenario.probability * cost
                except Exception as e:
                    print(f"Scenario {scenario.id} failed: {e}")
                    self.scenario_costs[scenario.id] = float('inf')
        
        self.expected_cost = expected_cost
        return expected_cost
    
    def _solve_single_scenario(
        self,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int],
        scenario: DemandScenario,
        solver: Optional[str],
        time_limit: int
    ) -> float:
        """Solve MILP for a single scenario."""
        milp = Stage2MILP(
            network=self.network,
            config=self.config,
            x_assign=x_assign,
            y_assign=y_assign,
            scenario=scenario
        )
        
        milp.build_model()
        cost = milp.solve(solver=solver, time_limit=time_limit)
        
        # Store solution
        if milp.solve_status == 'Optimal':
            self.scenario_solutions[scenario.id] = milp.get_solution()
        
        return cost
    
    def get_cost_statistics(self) -> Dict[str, float]:
        """
        Get statistics about scenario costs.
        
        Returns:
            Dict with mean, std, min, max, CVaR statistics
        """
        if not self.scenario_costs:
            return {}
        
        costs = list(self.scenario_costs.values())
        probs = [s.probability for s in self.scenarios]
        
        # Weighted statistics
        mean_cost = sum(c * p for c, p in zip(costs, probs))
        
        # Variance
        variance = sum(p * (c - mean_cost)**2 for c, p in zip(costs, probs))
        std_cost = np.sqrt(variance)
        
        # CVaR at 95% (worst 5% of scenarios)
        sorted_costs = sorted(zip(costs, probs), key=lambda x: -x[0])
        cvar_threshold = 0.05
        cvar_sum = 0
        cum_prob = 0
        for cost, prob in sorted_costs:
            if cum_prob < cvar_threshold:
                take_prob = min(prob, cvar_threshold - cum_prob)
                cvar_sum += cost * take_prob
                cum_prob += take_prob
        cvar = cvar_sum / cvar_threshold if cvar_threshold > 0 else costs[0]
        
        return {
            "expected_cost": mean_cost,
            "std_cost": std_cost,
            "min_cost": min(costs),
            "max_cost": max(costs),
            "cvar_95": cvar
        }
    
    def get_aggregate_flows(self) -> Dict[str, np.ndarray]:
        """
        Get expected flows across scenarios.
        
        Returns:
            Dict with expected flow matrices
        """
        if not self.scenario_solutions:
            return {}
        
        # Aggregate flows weighted by probability
        num_s = self.network.num_suppliers
        num_w = self.network.num_warehouses
        num_c = self.network.num_customers
        H = self.config.scenarios.horizon_days
        
        expected_f = np.zeros((num_s, num_w, H))
        expected_g = np.zeros((num_w, num_c, H))
        
        for scenario in self.scenarios:
            if scenario.id in self.scenario_solutions:
                sol = self.scenario_solutions[scenario.id]
                prob = scenario.probability
                
                for (i, j, t), val in sol.get("f", {}).items():
                    expected_f[i, j, t] += prob * val
                
                for (j, k, t), val in sol.get("g", {}).items():
                    expected_g[j, k, t] += prob * val
        
        return {
            "expected_flow_sw": expected_f,
            "expected_flow_wc": expected_g
        }
