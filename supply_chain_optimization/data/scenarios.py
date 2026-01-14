"""
Demand scenario generator for stochastic optimization.

Generates d_{k,t}^ω: demand of customer k on day t under scenario ω.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from .models import SupplyChainNetwork
from ..config import ScenarioConfig


@dataclass
class DemandScenario:
    """
    A single demand scenario ω.
    
    Attributes:
        id: Scenario identifier
        probability: π_ω - probability of this scenario
        demand: d_{k,t}^ω - demand matrix (num_customers x horizon_days)
    """
    id: int
    probability: float
    demand: np.ndarray  # Shape: (num_customers, horizon_days)
    
    def get_demand(self, customer_k: int, day_t: int) -> float:
        """Get demand d_{k,t}^ω for customer k on day t."""
        return self.demand[customer_k, day_t]
    
    def get_total_demand(self, customer_k: int) -> float:
        """Get total demand for customer k across all days."""
        return np.sum(self.demand[customer_k, :])


def generate_demand_scenarios(
    network: SupplyChainNetwork,
    config: ScenarioConfig,
    random_seed: Optional[int] = None
) -> List[DemandScenario]:
    """
    Generate stochastic demand scenarios.
    
    Creates Ω scenarios with varying demand patterns:
    - Base scenario: Expected demand
    - High demand scenarios: 1.2x - 1.5x expected
    - Low demand scenarios: 0.7x - 0.9x expected
    
    Args:
        network: The supply chain network
        config: Scenario configuration
        random_seed: Random seed for reproducibility
        
    Returns:
        List of DemandScenario objects with probabilities summing to 1
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    num_scenarios = config.num_scenarios
    num_customers = network.num_customers
    horizon = config.horizon_days
    
    scenarios = []
    
    # Scenario probabilities (equal by default, but can be customized)
    probabilities = _generate_scenario_probabilities(num_scenarios)
    
    for omega in range(num_scenarios):
        # Generate demand multiplier for this scenario
        multiplier = _get_scenario_multiplier(omega, num_scenarios)
        
        # Generate demand matrix
        demand = np.zeros((num_customers, horizon))
        
        for k, customer in enumerate(network.customers):
            if customer.is_factory:
                # Factory has more stable, higher demand
                base_demand = customer.average_demand
                noise_factor = 0.1  # Less variance for factory
            else:
                base_demand = customer.average_demand
                noise_factor = config.demand_std_factor
            
            # Generate daily demand with some autocorrelation
            demand[k, :] = _generate_correlated_demand(
                base_demand=base_demand * multiplier,
                std_factor=noise_factor,
                num_days=horizon
            )
        
        scenarios.append(DemandScenario(
            id=omega,
            probability=probabilities[omega],
            demand=demand
        ))
    
    return scenarios


def _generate_scenario_probabilities(num_scenarios: int) -> List[float]:
    """
    Generate scenario probabilities π_ω.
    
    Uses a distribution that gives more weight to base scenarios:
    - Central scenarios get higher probability
    - Extreme scenarios get lower probability
    """
    if num_scenarios == 1:
        return [1.0]
    
    if num_scenarios == 2:
        return [0.6, 0.4]  # Base scenario slightly more likely
    
    if num_scenarios == 3:
        return [0.5, 0.3, 0.2]  # Base, high, low
    
    # For more scenarios, use a triangular-like distribution
    weights = []
    mid = (num_scenarios - 1) / 2
    for i in range(num_scenarios):
        weight = 1.0 - abs(i - mid) / (mid + 1) * 0.5
        weights.append(weight)
    
    # Normalize to sum to 1
    total = sum(weights)
    return [w / total for w in weights]


def _get_scenario_multiplier(omega: int, num_scenarios: int) -> float:
    """
    Get demand multiplier for scenario omega.
    
    Returns:
        Multiplier (1.0 = base, >1.0 = high demand, <1.0 = low demand)
    """
    if num_scenarios == 1:
        return 1.0
    
    if num_scenarios == 2:
        return [1.0, 1.3][omega]  # Base, high
    
    if num_scenarios == 3:
        return [1.0, 1.3, 0.7][omega]  # Base, high, low
    
    # For more scenarios, spread evenly from 0.6 to 1.4
    return 0.6 + (omega / (num_scenarios - 1)) * 0.8


def _generate_correlated_demand(
    base_demand: float,
    std_factor: float,
    num_days: int,
    autocorr: float = 0.3
) -> np.ndarray:
    """
    Generate temporally correlated demand values.
    
    Uses AR(1) process for realistic demand patterns.
    
    Args:
        base_demand: Mean demand level
        std_factor: Standard deviation as fraction of mean
        num_days: Number of days to generate
        autocorr: Autocorrelation coefficient
        
    Returns:
        Array of daily demand values (non-negative)
    """
    std = base_demand * std_factor
    
    # Generate AR(1) noise
    noise = np.zeros(num_days)
    innovation_std = std * np.sqrt(1 - autocorr**2)
    noise[0] = np.random.normal(0, std)
    
    for t in range(1, num_days):
        noise[t] = autocorr * noise[t-1] + np.random.normal(0, innovation_std)
    
    # Add to base demand, ensure non-negative
    demand = np.maximum(0, base_demand + noise)
    
    return demand


def compute_average_demand(scenarios: List[DemandScenario]) -> np.ndarray:
    """
    Compute expected demand E[d_{k,t}] across all scenarios.
    
    Args:
        scenarios: List of demand scenarios
        
    Returns:
        Expected demand matrix (num_customers x horizon_days)
    """
    expected = np.zeros_like(scenarios[0].demand)
    
    for scenario in scenarios:
        expected += scenario.probability * scenario.demand
    
    return expected


def compute_demand_variance(scenarios: List[DemandScenario]) -> np.ndarray:
    """
    Compute demand variance Var[d_{k,t}] across scenarios.
    
    Args:
        scenarios: List of demand scenarios
        
    Returns:
        Variance matrix (num_customers x horizon_days)
    """
    expected = compute_average_demand(scenarios)
    
    variance = np.zeros_like(expected)
    for scenario in scenarios:
        variance += scenario.probability * (scenario.demand - expected)**2
    
    return variance


def compute_safety_stock(
    network: SupplyChainNetwork,
    scenarios: List[DemandScenario],
    warehouse_idx: int
) -> Dict[int, float]:
    """
    Compute safety stock for each customer served by a warehouse.
    
    Safety_{j,k} = z_k × σ_{k} × √(L^WC_{j,k})
    
    Args:
        network: The supply chain network
        scenarios: Demand scenarios
        warehouse_idx: Warehouse index j
        
    Returns:
        Dictionary mapping customer k to safety stock requirement
    """
    if network.lead_time_wc is None:
        from .distances import compute_lead_time_matrices
        compute_lead_time_matrices(network)
    
    demand_variance = compute_demand_variance(scenarios)
    safety_stock = {}
    
    for k, customer in enumerate(network.customers):
        # Standard deviation of demand
        sigma_k = np.sqrt(np.mean(demand_variance[k, :]))
        
        # Lead time from this warehouse to customer
        lead_time = network.lead_time_wc[warehouse_idx, k]
        
        # Safety stock formula
        z_k = customer.service_level_z
        safety_stock[k] = z_k * sigma_k * np.sqrt(lead_time)
    
    return safety_stock
