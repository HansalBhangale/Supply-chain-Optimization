"""
Configuration module for supply chain optimization.

Contains all problem parameters, network sizes, costs, and QAOA settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class NetworkConfig:
    """Network topology configuration."""
    num_suppliers: int = 10
    num_warehouses: int = 5
    num_customers: int = 20  # Including factory at index 0
    factory_index: int = 0


@dataclass
class CostConfig:
    """Cost parameters for the supply chain."""
    fixed_cost_per_truck: float = 100.0  # cFix: Fixed cost per truck dispatched
    variable_cost_per_km: float = 0.5    # cVar: Variable cost per unit-km
    holding_cost_default: float = 1.0    # h_j: Default holding cost per unit per day
    shortage_penalty_default: float = 50.0  # p_k: Default shortage penalty
    factory_shortage_penalty: float = 10000.0  # p_0: Factory has huge penalty


@dataclass
class CapacityConfig:
    """Capacity constraints."""
    supplier_capacity_default: float = 1000.0  # capSup: Units per day
    warehouse_capacity_default: float = 5000.0  # capWh: Storage capacity
    truck_capacity: float = 100.0  # capTruck: Units per truck


@dataclass
class LeadTimeConfig:
    """Lead time settings (in days)."""
    supplier_to_warehouse_default: int = 2  # L^SW
    warehouse_to_customer_default: int = 1  # L^WC


@dataclass
class ScenarioConfig:
    """Stochastic scenario configuration."""
    num_scenarios: int = 3  # |Ω|: Number of demand scenarios
    horizon_days: int = 28  # H: Time horizon (4 weeks)
    base_demand_mean: float = 50.0  # Average daily demand per customer
    demand_std_factor: float = 0.3  # Demand standard deviation factor


@dataclass
class QAOAConfig:
    """QAOA algorithm configuration."""
    depth: int = 2  # p: QAOA circuit depth
    shots: int = 1024  # Number of measurement shots
    optimizer: str = "COBYLA"  # Classical optimizer
    maxiter: int = 100  # Maximum optimizer iterations
    use_warm_start: bool = True  # Warm-start from previous solution


@dataclass
class PenaltyConfig:
    """Penalty weights for QUBO constraints (Section 5: Penalty Dominance)."""
    margin: float = 0.1  # Safety margin for penalty calculation
    vendor_single_sourcing: float = 0.0  # λ₁: Auto-calculated if 0
    factory_redundancy: float = 0.0  # λ₂: Auto-calculated if 0
    capacity_proxy: float = 0.0  # λ₃: Auto-calculated if 0
    supplier_connectivity: float = 0.0  # λ₄: Auto-calculated if 0
    factory_redundancy_count: int = 2  # Number of warehouses for factory


@dataclass
class BigMConfig:
    """Big-M constants for MILP constraints."""
    max_trucks_per_day: int = 5  # Maximum trucks per route per day
    # M^SW = max_trucks_per_day × horizon_days


@dataclass 
class SafetyStockConfig:
    """Safety stock configuration by customer tier."""
    factory_z_score: float = 3.0  # z_0: Very high service level
    tier1_z_score: float = 2.33  # ~99% service level
    tier2_z_score: float = 1.96  # ~97.5% service level
    tier3_z_score: float = 1.64  # ~95% service level


@dataclass
class SupplyChainConfig:
    """Master configuration for the entire optimization problem."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    lead_times: LeadTimeConfig = field(default_factory=LeadTimeConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    qaoa: QAOAConfig = field(default_factory=QAOAConfig)
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)
    big_m: BigMConfig = field(default_factory=BigMConfig)
    safety_stock: SafetyStockConfig = field(default_factory=SafetyStockConfig)
    
    # Random seed for reproducibility
    random_seed: Optional[int] = 42
    
    def __post_init__(self):
        """Set random seed if provided."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def get_big_m_sw(self) -> int:
        """Calculate Big-M for supplier-warehouse constraints."""
        return self.big_m.max_trucks_per_day * self.scenarios.horizon_days
    
    def get_big_m_wc(self) -> int:
        """Calculate Big-M for warehouse-customer constraints."""
        return self.big_m.max_trucks_per_day * self.scenarios.horizon_days


def create_default_config() -> SupplyChainConfig:
    """Create a default configuration for testing."""
    return SupplyChainConfig()


def create_small_config() -> SupplyChainConfig:
    """Create a small-scale configuration for quick testing."""
    return SupplyChainConfig(
        network=NetworkConfig(
            num_suppliers=5,
            num_warehouses=3,
            num_customers=10,
        ),
        scenarios=ScenarioConfig(
            num_scenarios=2,
            horizon_days=7,
        ),
        qaoa=QAOAConfig(
            depth=1,
            shots=512,
            maxiter=50,
        ),
    )
