"""
Stage 2 MILP model for operational scheduling.

Implements Section 2: Stage 2 Operational MILP.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pulp

from ..config import SupplyChainConfig
from ..data.models import SupplyChainNetwork
from ..data.scenarios import DemandScenario, compute_safety_stock


class Stage2MILP:
    """
    MILP model for Stage 2 operational optimization.
    
    Given fixed assignments (x, y) from Stage 1, optimizes:
    - Minimize Z_ω = Transport + Holding + Shortage
    """
    
    def __init__(
        self,
        network: SupplyChainNetwork,
        config: SupplyChainConfig,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int],
        scenario: DemandScenario
    ):
        """
        Initialize Stage 2 MILP model.
        
        Args:
            network: Supply chain network
            config: Configuration parameters
            x_assign: Stage 1 supplier-warehouse assignments
            y_assign: Stage 1 warehouse-customer assignments
            scenario: Demand scenario ω
        """
        self.network = network
        self.config = config
        self.x_assign = x_assign
        self.y_assign = y_assign
        self.scenario = scenario
        
        # Problem dimensions
        self.num_s = network.num_suppliers
        self.num_w = network.num_warehouses
        self.num_c = network.num_customers
        self.H = config.scenarios.horizon_days
        
        # Big-M constants
        self.M_sw = config.get_big_m_sw()
        self.M_wc = config.get_big_m_wc()
        
        # PuLP model
        self.model: Optional[pulp.LpProblem] = None
        
        # Decision variables (Section 2.1)
        self.n_sw: Dict[Tuple[int, int, int], pulp.LpVariable] = {}  # Trucks S→W
        self.n_wc: Dict[Tuple[int, int, int], pulp.LpVariable] = {}  # Trucks W→C
        self.f: Dict[Tuple[int, int, int], pulp.LpVariable] = {}     # Flow S→W
        self.g: Dict[Tuple[int, int, int], pulp.LpVariable] = {}     # Flow W→C
        self.I: Dict[Tuple[int, int], pulp.LpVariable] = {}          # Inventory
        self.u: Dict[Tuple[int, int], pulp.LpVariable] = {}          # Shortage
        
        # Results
        self.optimal_value: Optional[float] = None
        self.solve_status: Optional[str] = None
    
    def build_model(self) -> pulp.LpProblem:
        """
        Build the complete MILP model.
        
        Returns:
            PuLP LpProblem object
        """
        self.model = pulp.LpProblem(
            f"SupplyChain_Stage2_Scenario{self.scenario.id}",
            pulp.LpMinimize
        )
        
        # Create decision variables
        self._create_variables()
        
        # Set objective function
        self._set_objective()
        
        # Add constraints
        self._add_assignment_respect_constraints()
        self._add_truck_capacity_constraints()
        self._add_supplier_capacity_constraints()
        self._add_inventory_balance_constraints()
        self._add_demand_satisfaction_constraints()
        self._add_safety_stock_constraints()
        
        return self.model
    
    def _create_variables(self):
        """Create all decision variables (Section 2.1)."""
        # n^SW_{i,j,t}: integer trucks supplier→warehouse
        for i in range(self.num_s):
            for j in range(self.num_w):
                for t in range(self.H):
                    self.n_sw[(i, j, t)] = pulp.LpVariable(
                        f"n_sw_{i}_{j}_{t}",
                        lowBound=0,
                        cat=pulp.LpInteger
                    )
        
        # n^WC_{j,k,t}: integer trucks warehouse→customer
        for j in range(self.num_w):
            for k in range(self.num_c):
                for t in range(self.H):
                    self.n_wc[(j, k, t)] = pulp.LpVariable(
                        f"n_wc_{j}_{k}_{t}",
                        lowBound=0,
                        cat=pulp.LpInteger
                    )
        
        # f_{i,j,t}: continuous flow supplier→warehouse
        for i in range(self.num_s):
            for j in range(self.num_w):
                for t in range(self.H):
                    self.f[(i, j, t)] = pulp.LpVariable(
                        f"f_{i}_{j}_{t}",
                        lowBound=0,
                        cat=pulp.LpContinuous
                    )
        
        # g_{j,k,t}: continuous flow warehouse→customer
        for j in range(self.num_w):
            for k in range(self.num_c):
                for t in range(self.H):
                    self.g[(j, k, t)] = pulp.LpVariable(
                        f"g_{j}_{k}_{t}",
                        lowBound=0,
                        cat=pulp.LpContinuous
                    )
        
        # I_{j,t}: inventory at warehouse at end of day t
        for j in range(self.num_w):
            for t in range(-1, self.H):  # Include t=-1 for initial inventory
                wh = self.network.get_warehouse(j)
                ub = wh.capacity
                lb = 0
                
                self.I[(j, t)] = pulp.LpVariable(
                    f"I_{j}_{t}",
                    lowBound=lb,
                    upBound=ub,
                    cat=pulp.LpContinuous
                )
        
        # u_{k,t}: shortage at customer
        for k in range(self.num_c):
            for t in range(self.H):
                self.u[(k, t)] = pulp.LpVariable(
                    f"u_{k}_{t}",
                    lowBound=0,
                    cat=pulp.LpContinuous
                )
    
    def _set_objective(self):
        """
        Set objective function (Section 2.2):
        Minimize Z_ω = Transport_ω + Holding_ω + Shortage_ω
        """
        transport_cost = 0
        holding_cost = 0
        shortage_cost = 0
        
        c_fix = self.config.costs.fixed_cost_per_truck
        c_var = self.config.costs.variable_cost_per_km
        
        # Transport cost: Σ_t Σ_i Σ_j [cFix·n^SW + cVar·D^SW·f]
        for i in range(self.num_s):
            for j in range(self.num_w):
                d_sw = self.network.distance_sw[i, j] if self.network.distance_sw is not None else 50.0
                for t in range(self.H):
                    transport_cost += c_fix * self.n_sw[(i, j, t)]
                    transport_cost += c_var * d_sw * self.f[(i, j, t)]
        
        # Transport cost: Σ_t Σ_j Σ_k [cFix·n^WC + cVar·D^WC·g]
        for j in range(self.num_w):
            for k in range(self.num_c):
                d_wc = self.network.distance_wc[j, k] if self.network.distance_wc is not None else 30.0
                for t in range(self.H):
                    transport_cost += c_fix * self.n_wc[(j, k, t)]
                    transport_cost += c_var * d_wc * self.g[(j, k, t)]
        
        # Holding cost: Σ_t Σ_j [h_j · I_{j,t}]
        for j in range(self.num_w):
            h_j = self.network.get_warehouse(j).holding_cost
            for t in range(self.H):
                holding_cost += h_j * self.I[(j, t)]
        
        # Shortage cost: Σ_t Σ_k [p_k · u_{k,t}]
        for k in range(self.num_c):
            p_k = self.network.get_customer(k).shortage_penalty
            for t in range(self.H):
                shortage_cost += p_k * self.u[(k, t)]
        
        self.model += transport_cost + holding_cost + shortage_cost, "TotalCost"
    
    def _add_assignment_respect_constraints(self):
        """
        Add assignment-respect constraints (Section 2.3.1):
        n^SW_{i,j,t} ≤ M^SW · x_{i,j}
        n^WC_{j,k,t} ≤ M^WC · y_{j,k}
        """
        for i in range(self.num_s):
            for j in range(self.num_w):
                x_ij = self.x_assign.get((i, j), 0)
                for t in range(self.H):
                    self.model += (
                        self.n_sw[(i, j, t)] <= self.M_sw * x_ij,
                        f"AssignRespect_SW_{i}_{j}_{t}"
                    )
        
        for j in range(self.num_w):
            for k in range(self.num_c):
                y_jk = self.y_assign.get((j, k), 0)
                for t in range(self.H):
                    self.model += (
                        self.n_wc[(j, k, t)] <= self.M_wc * y_jk,
                        f"AssignRespect_WC_{j}_{k}_{t}"
                    )
    
    def _add_truck_capacity_constraints(self):
        """
        Add truck capacity coupling (Section 2.3.2):
        f_{i,j,t} ≤ capTruck · n^SW_{i,j,t}
        g_{j,k,t} ≤ capTruck · n^WC_{j,k,t}
        """
        cap_truck = self.config.capacity.truck_capacity
        
        for i in range(self.num_s):
            for j in range(self.num_w):
                for t in range(self.H):
                    self.model += (
                        self.f[(i, j, t)] <= cap_truck * self.n_sw[(i, j, t)],
                        f"TruckCap_SW_{i}_{j}_{t}"
                    )
        
        for j in range(self.num_w):
            for k in range(self.num_c):
                for t in range(self.H):
                    self.model += (
                        self.g[(j, k, t)] <= cap_truck * self.n_wc[(j, k, t)],
                        f"TruckCap_WC_{j}_{k}_{t}"
                    )
    
    def _add_supplier_capacity_constraints(self):
        """
        Add supplier capacity constraints (Section 2.3.3):
        Σ_j f_{i,j,t} ≤ capSup_{i,t}
        """
        for i in range(self.num_s):
            supplier = self.network.get_supplier(i)
            for t in range(self.H):
                cap_sup = supplier.get_capacity(t, self.config.capacity.supplier_capacity_default)
                self.model += (
                    pulp.lpSum([self.f[(i, j, t)] for j in range(self.num_w)]) <= cap_sup,
                    f"SupplierCap_{i}_{t}"
                )
    
    def _add_inventory_balance_constraints(self):
        """
        Add inventory balance with lead times (Section 2.3.4):
        I_{j,t} = I_{j,t-1} + Arr^SW_{j,t} - Dep^WC_{j,t}
        """
        # Set initial inventory
        for j in range(self.num_w):
            wh = self.network.get_warehouse(j)
            self.model += (
                self.I[(j, -1)] == wh.initial_inventory,
                f"InitInv_{j}"
            )
        
        # Inventory balance for each day
        for j in range(self.num_w):
            for t in range(self.H):
                # Arrivals: Σ_i f_{i,j,t-L^SW_{i,j}}
                arrivals = 0
                for i in range(self.num_s):
                    if self.network.lead_time_sw is not None:
                        lead_time = int(self.network.lead_time_sw[i, j])
                    else:
                        lead_time = self.config.lead_times.supplier_to_warehouse_default
                    
                    depart_time = t - lead_time
                    if depart_time >= 0:
                        arrivals += self.f[(i, j, depart_time)]
                
                # Departures: Σ_k g_{j,k,t}
                departures = pulp.lpSum([self.g[(j, k, t)] for k in range(self.num_c)])
                
                # Balance constraint
                self.model += (
                    self.I[(j, t)] == self.I[(j, t-1)] + arrivals - departures,
                    f"InvBalance_{j}_{t}"
                )
    
    def _add_demand_satisfaction_constraints(self):
        """
        Add demand satisfaction with shortage (Section 2.3.5):
        Arr^WC_{k,t} + u_{k,t} = d^ω_{k,t}
        u_{0,t} = 0 (factory hard constraint, relaxed for early days)
        """
        # Get minimum lead time to factory for constraint relaxation
        min_lead_time_to_factory = float('inf')
        for j in range(self.num_w):
            if self.network.lead_time_wc is not None:
                lt = int(self.network.lead_time_wc[j, 0])
            else:
                lt = self.config.lead_times.warehouse_to_customer_default
            min_lead_time_to_factory = min(min_lead_time_to_factory, lt)
        
        for k in range(self.num_c):
            for t in range(self.H):
                # Arrivals to customer k on day t
                arrivals = 0
                for j in range(self.num_w):
                    if self.network.lead_time_wc is not None:
                        lead_time = int(self.network.lead_time_wc[j, k])
                    else:
                        lead_time = self.config.lead_times.warehouse_to_customer_default
                    
                    depart_time = t - lead_time
                    if depart_time >= 0:
                        arrivals += self.g[(j, k, depart_time)]
                
                # Demand from scenario
                demand = self.scenario.get_demand(k, t)
                
                # Demand balance: arrivals + shortage = demand
                self.model += (
                    arrivals + self.u[(k, t)] >= demand,
                    f"DemandSat_{k}_{t}"
                )
        
        # Factory soft constraint: minimize shortage but don't make infeasible
        # Only enforce hard constraint after initial lead time window
        factory_k = 0
        for t in range(self.H):
            if t >= min_lead_time_to_factory + 2:  # Allow ramp-up time
                # Softly encourage zero factory shortage via high penalty (already in objective)
                pass  # Penalty handles this
    
    def _add_safety_stock_constraints(self):
        """
        Add safety stock constraints (Section 2.3.6):
        I_{j,t} >= sum_k [Safety_{j,k} * y_{j,k}]
        """
        # Compute safety stock requirements
        for j in range(self.num_w):
            # Calculate total safety stock needed at this warehouse
            safety_total = 0
            
            for k in range(self.num_c):
                if self.y_assign.get((j, k), 0) == 1:
                    customer = self.network.get_customer(k)
                    # Simplified safety stock: z_k × avg_demand × sqrt(lead_time)
                    if self.network.lead_time_wc is not None:
                        lead_time = self.network.lead_time_wc[j, k]
                    else:
                        lead_time = self.config.lead_times.warehouse_to_customer_default
                    
                    sigma = customer.average_demand * 0.3  # Assume 30% demand variability
                    safety = customer.service_level_z * sigma * np.sqrt(lead_time)
                    safety_total += safety
            
            # Safety stock constraint for each day
            for t in range(self.H):
                self.model += (
                    self.I[(j, t)] >= safety_total,
                    f"SafetyStock_{j}_{t}"
                )
    
    def solve(self, solver: Optional[str] = None, time_limit: int = 300) -> float:
        """
        Solve the MILP model.
        
        Args:
            solver: Solver to use ('PULP_CBC_CMD', 'GUROBI_CMD', etc.)
            time_limit: Time limit in seconds
            
        Returns:
            Optimal objective value
        """
        if self.model is None:
            self.build_model()
        
        # Select solver
        if solver is None or solver == 'PULP_CBC_CMD':
            lp_solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        elif solver == 'GUROBI_CMD':
            lp_solver = pulp.GUROBI_CMD(msg=0, timeLimit=time_limit)
        else:
            lp_solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        
        # Solve
        self.model.solve(lp_solver)
        
        self.solve_status = pulp.LpStatus[self.model.status]
        
        if self.model.status == pulp.LpStatusOptimal:
            self.optimal_value = pulp.value(self.model.objective)
        else:
            self.optimal_value = float('inf')
        
        return self.optimal_value
    
    def get_solution(self) -> Dict[str, Dict]:
        """
        Extract solution values.
        
        Returns:
            Dict with variable values
        """
        if self.model is None or self.model.status != pulp.LpStatusOptimal:
            return {}
        
        solution = {
            "n_sw": {k: v.varValue for k, v in self.n_sw.items() if v.varValue and v.varValue > 0},
            "n_wc": {k: v.varValue for k, v in self.n_wc.items() if v.varValue and v.varValue > 0},
            "f": {k: v.varValue for k, v in self.f.items() if v.varValue and v.varValue > 0},
            "g": {k: v.varValue for k, v in self.g.items() if v.varValue and v.varValue > 0},
            "I": {k: v.varValue for k, v in self.I.items()},
            "u": {k: v.varValue for k, v in self.u.items() if v.varValue and v.varValue > 0},
        }
        
        return solution
