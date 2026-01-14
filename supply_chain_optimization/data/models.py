"""
Data models for supply chain entities.

Defines Supplier, Warehouse, Customer, and the complete network structure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class Supplier:
    """
    Supplier entity (index i in the mathematical formulation).
    
    Attributes:
        id: Unique supplier identifier
        name: Human-readable name
        location: (x, y) coordinates for distance calculation
        capacity: capSup_{i,t} - capacity available per time period
    """
    id: int
    name: str
    location: Tuple[float, float]
    capacity: Dict[int, float] = field(default_factory=dict)
    
    def get_capacity(self, t: int, default: float = 1000.0) -> float:
        """Get capacity for time period t."""
        return self.capacity.get(t, default)


@dataclass
class Warehouse:
    """
    Warehouse entity (index j in the mathematical formulation).
    
    Attributes:
        id: Unique warehouse identifier
        name: Human-readable name
        location: (x, y) coordinates
        capacity: capWh_j - storage capacity
        holding_cost: h_j - holding cost per unit per day
        initial_inventory: I^init_j - starting inventory
    """
    id: int
    name: str
    location: Tuple[float, float]
    capacity: float = 5000.0
    holding_cost: float = 1.0
    initial_inventory: float = 0.0


@dataclass
class Customer:
    r"""
    Customer entity (index k in the mathematical formulation).
    Factory is k=0, vendors are V = C \ {0}.
    
    Attributes:
        id: Unique customer identifier
        name: Human-readable name
        location: (x, y) coordinates
        is_factory: True if this is the factory (k=0)
        shortage_penalty: p_k - penalty per unit shortage
        service_level_z: z_k - z-score for service level (safety stock)
        tier: Customer tier (1, 2, or 3) for prioritization
        average_demand: Average daily demand for capacity proxy calculation
    """
    id: int
    name: str
    location: Tuple[float, float]
    is_factory: bool = False
    shortage_penalty: float = 50.0
    service_level_z: float = 1.96
    tier: int = 2
    average_demand: float = 50.0


@dataclass
class SupplyChainNetwork:
    r"""
    Complete supply chain network containing all entities.
    
    Notation from mathematical formulation:
        S: suppliers, index i
        W: warehouses, index j  
        C: customers (factory + vendors), index k
        V: vendors only, V = C \ {0}
    """
    suppliers: List[Supplier]
    warehouses: List[Warehouse]
    customers: List[Customer]
    
    # Distance matrices (D^SW and D^WC)
    distance_sw: Optional[np.ndarray] = None  # |S| x |W|
    distance_wc: Optional[np.ndarray] = None  # |W| x |C|
    
    # Lead time matrices (L^SW and L^WC)
    lead_time_sw: Optional[np.ndarray] = None  # |S| x |W|
    lead_time_wc: Optional[np.ndarray] = None  # |W| x |C|
    
    def __post_init__(self):
        """Validate the network structure."""
        # Ensure factory is at index 0
        if self.customers and not self.customers[0].is_factory:
            raise ValueError("Customer at index 0 must be the factory")
    
    @property
    def num_suppliers(self) -> int:
        """Number of suppliers |S|."""
        return len(self.suppliers)
    
    @property
    def num_warehouses(self) -> int:
        """Number of warehouses |W|."""
        return len(self.warehouses)
    
    @property
    def num_customers(self) -> int:
        """Number of customers |C| (including factory)."""
        return len(self.customers)
    
    @property
    def num_vendors(self) -> int:
        """Number of vendors |V| = |C| - 1."""
        return len(self.customers) - 1
    
    @property
    def factory(self) -> Customer:
        """Get the factory customer (k=0)."""
        return self.customers[0]
    
    @property
    def vendors(self) -> List[Customer]:
        r"""Get vendor customers (V = C \ {0})."""
        return self.customers[1:]
    
    def get_supplier(self, i: int) -> Supplier:
        """Get supplier by index."""
        return self.suppliers[i]
    
    def get_warehouse(self, j: int) -> Warehouse:
        """Get warehouse by index."""
        return self.warehouses[j]
    
    def get_customer(self, k: int) -> Customer:
        """Get customer by index."""
        return self.customers[k]


def create_sample_network(
    num_suppliers: int = 10,
    num_warehouses: int = 5,
    num_customers: int = 20,
    grid_size: float = 100.0,
    random_seed: Optional[int] = 42
) -> SupplyChainNetwork:
    """
    Create a sample supply chain network with random locations.
    
    Args:
        num_suppliers: Number of suppliers
        num_warehouses: Number of warehouses
        num_customers: Number of customers (including factory at index 0)
        grid_size: Size of the coordinate grid
        random_seed: Random seed for reproducibility
        
    Returns:
        SupplyChainNetwork with randomly placed entities
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create suppliers (clustered in one region)
    suppliers = []
    for i in range(num_suppliers):
        suppliers.append(Supplier(
            id=i,
            name=f"Supplier_{i}",
            location=(
                np.random.uniform(0, grid_size * 0.3),
                np.random.uniform(grid_size * 0.3, grid_size * 0.7)
            ),
            capacity={t: np.random.uniform(800, 1200) for t in range(28)}
        ))
    
    # Create warehouses (spread across middle region)
    warehouses = []
    for j in range(num_warehouses):
        warehouses.append(Warehouse(
            id=j,
            name=f"Warehouse_{j}",
            location=(
                np.random.uniform(grid_size * 0.3, grid_size * 0.7),
                np.random.uniform(grid_size * 0.2, grid_size * 0.8)
            ),
            capacity=np.random.uniform(4000, 6000),
            holding_cost=np.random.uniform(0.8, 1.2),
            initial_inventory=np.random.uniform(500, 1500)
        ))
    
    # Create customers (factory at center, vendors spread across right region)
    customers = []
    
    # Factory at index 0 (center of the grid)
    customers.append(Customer(
        id=0,
        name="Factory",
        location=(grid_size * 0.5, grid_size * 0.5),
        is_factory=True,
        shortage_penalty=10000.0,  # Huge penalty for factory
        service_level_z=3.0,  # Very high service level
        tier=0,
        average_demand=200.0  # Factory has higher demand
    ))
    
    # Vendors
    for k in range(1, num_customers):
        tier = 1 if k <= num_customers // 4 else (2 if k <= num_customers // 2 else 3)
        z_scores = {1: 2.33, 2: 1.96, 3: 1.64}
        penalties = {1: 100.0, 2: 50.0, 3: 25.0}
        
        customers.append(Customer(
            id=k,
            name=f"Vendor_{k}",
            location=(
                np.random.uniform(grid_size * 0.6, grid_size),
                np.random.uniform(0, grid_size)
            ),
            is_factory=False,
            shortage_penalty=penalties[tier],
            service_level_z=z_scores[tier],
            tier=tier,
            average_demand=np.random.uniform(30, 70)
        ))
    
    return SupplyChainNetwork(
        suppliers=suppliers,
        warehouses=warehouses,
        customers=customers
    )
