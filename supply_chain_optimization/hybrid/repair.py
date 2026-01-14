"""
Post-QAOA feasibility repair algorithm.

Implements Section: Post-QAOA Feasibility Repair.
Ensures sampled QAOA bitstrings satisfy all constraints.
"""

from typing import Dict, Tuple, List, Set
import numpy as np

from ..data.models import SupplyChainNetwork
from ..config import SupplyChainConfig


class FeasibilityRepair:
    """
    Repairs QAOA solutions to ensure feasibility.
    
    Algorithm ensures:
    1. Vendor single-sourcing (exactly 1 warehouse per vendor)
    2. Factory redundancy (exactly 2 warehouses for factory)
    3. Supplier connectivity (each active warehouse has ≥1 supplier)
    4. Capacity compliance
    
    Complexity: O(|W||C| + |S||W|)
    """
    
    def __init__(
        self,
        network: SupplyChainNetwork,
        config: SupplyChainConfig
    ):
        """
        Initialize repair algorithm.
        
        Args:
            network: Supply chain network
            config: Configuration parameters
        """
        self.network = network
        self.config = config
        
        # Ensure distance matrices are computed
        if network.distance_wc is None:
            from ..data.distances import compute_distance_matrices
            compute_distance_matrices(network)
    
    def repair(
        self,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int]
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[str, int]]:
        """
        Repair assignments to ensure feasibility.
        
        Args:
            x_assign: Raw x_{i,j} assignments from QAOA
            y_assign: Raw y_{j,k} assignments from QAOA
            
        Returns:
            Tuple of (repaired x_assign, repaired y_assign, repair_stats)
        """
        # Copy assignments to avoid modifying originals
        x_new = dict(x_assign)
        y_new = dict(y_assign)
        
        stats = {
            "vendor_repairs": 0,
            "factory_repairs": 0,
            "connectivity_repairs": 0,
            "capacity_repairs": 0
        }
        
        # Step 1: Fix vendor single-sourcing
        vendor_repairs = self._repair_vendor_single_sourcing(y_new)
        stats["vendor_repairs"] = vendor_repairs
        
        # Step 2: Fix factory redundancy
        factory_repairs = self._repair_factory_redundancy(y_new)
        stats["factory_repairs"] = factory_repairs
        
        # Step 3: Ensure supplier connectivity
        connectivity_repairs = self._repair_supplier_connectivity(x_new, y_new)
        stats["connectivity_repairs"] = connectivity_repairs
        
        # Step 4: Check capacity and reassign if needed
        capacity_repairs = self._repair_capacity_violations(y_new)
        stats["capacity_repairs"] = capacity_repairs
        
        return x_new, y_new, stats
    
    def _repair_vendor_single_sourcing(
        self,
        y_assign: Dict[Tuple[int, int], int]
    ) -> int:
        """
        Ensure each vendor is served by exactly one warehouse.
        Keep assignment to nearest warehouse.
        
        Returns:
            Number of repairs made
        """
        repairs = 0
        num_w = self.network.num_warehouses
        
        # For each vendor (skip factory at k=0)
        for k in range(1, self.network.num_customers):
            assigned_warehouses = [
                j for j in range(num_w) 
                if y_assign.get((j, k), 0) == 1
            ]
            
            if len(assigned_warehouses) == 1:
                # Correct - exactly one warehouse
                continue
            elif len(assigned_warehouses) == 0:
                # No warehouse assigned - assign nearest
                nearest_j = self._get_nearest_warehouse(k)
                y_assign[(nearest_j, k)] = 1
                repairs += 1
            else:
                # Multiple warehouses - keep only nearest
                nearest_j = self._get_nearest_among(assigned_warehouses, k)
                for j in assigned_warehouses:
                    if j != nearest_j:
                        y_assign[(j, k)] = 0
                        repairs += 1
        
        return repairs
    
    def _repair_factory_redundancy(
        self,
        y_assign: Dict[Tuple[int, int], int],
        required_count: int = 2
    ) -> int:
        """
        Ensure factory is served by exactly 2 warehouses.
        
        Returns:
            Number of repairs made
        """
        repairs = 0
        factory_k = 0
        num_w = self.network.num_warehouses
        
        # Get current factory assignments
        assigned_warehouses = [
            j for j in range(num_w)
            if y_assign.get((j, factory_k), 0) == 1
        ]
        
        current_count = len(assigned_warehouses)
        
        if current_count == required_count:
            # Correct
            return 0
        elif current_count < required_count:
            # Need to add warehouses
            unassigned = [
                j for j in range(num_w)
                if j not in assigned_warehouses
            ]
            # Sort by distance to factory
            unassigned_sorted = sorted(
                unassigned,
                key=lambda j: self.network.distance_wc[j, factory_k]
            )
            # Add closest unassigned warehouses
            for j in unassigned_sorted[:required_count - current_count]:
                y_assign[(j, factory_k)] = 1
                repairs += 1
        else:
            # Need to remove warehouses - keep 2 nearest
            sorted_assigned = sorted(
                assigned_warehouses,
                key=lambda j: self.network.distance_wc[j, factory_k]
            )
            for j in sorted_assigned[required_count:]:
                y_assign[(j, factory_k)] = 0
                repairs += 1
        
        return repairs
    
    def _repair_supplier_connectivity(
        self,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int]
    ) -> int:
        """
        Ensure each warehouse with customers has at least one supplier.
        
        Returns:
            Number of repairs made
        """
        repairs = 0
        num_s = self.network.num_suppliers
        num_w = self.network.num_warehouses
        num_c = self.network.num_customers
        
        for j in range(num_w):
            # Check if warehouse has any customers
            has_customers = any(
                y_assign.get((j, k), 0) == 1
                for k in range(num_c)
            )
            
            if not has_customers:
                continue
            
            # Check if warehouse has any suppliers
            has_suppliers = any(
                x_assign.get((i, j), 0) == 1
                for i in range(num_s)
            )
            
            if not has_suppliers:
                # Assign nearest supplier
                nearest_i = self._get_nearest_supplier(j)
                x_assign[(nearest_i, j)] = 1
                repairs += 1
        
        return repairs
    
    def _repair_capacity_violations(
        self,
        y_assign: Dict[Tuple[int, int], int]
    ) -> int:
        """
        Check capacity and reassign lowest-priority customers if violated.
        
        Returns:
            Number of repairs made
        """
        repairs = 0
        num_w = self.network.num_warehouses
        num_c = self.network.num_customers
        
        for j in range(num_w):
            warehouse = self.network.get_warehouse(j)
            capacity = warehouse.capacity
            
            # Calculate current demand load
            assigned_customers = [
                k for k in range(num_c)
                if y_assign.get((j, k), 0) == 1
            ]
            
            total_demand = sum(
                self.network.get_customer(k).average_demand
                for k in assigned_customers
            )
            
            # Check if over capacity (simplified check)
            capacity_threshold = capacity * 0.9  # 90% utilization target
            
            if total_demand > capacity_threshold:
                # Remove lowest-priority customers (highest tier number)
                customers_by_priority = sorted(
                    assigned_customers,
                    key=lambda k: (
                        -self.network.get_customer(k).tier,  # Lower tier = higher priority
                        self.network.get_customer(k).is_factory  # Factory has highest priority
                    )
                )
                
                # Reassign customers until under capacity
                for k in customers_by_priority:
                    if self.network.get_customer(k).is_factory:
                        continue  # Never remove factory
                    
                    if total_demand <= capacity_threshold:
                        break
                    
                    # Find alternative warehouse
                    alt_j = self._find_alternative_warehouse(k, j, y_assign)
                    if alt_j is not None:
                        y_assign[(j, k)] = 0
                        y_assign[(alt_j, k)] = 1
                        total_demand -= self.network.get_customer(k).average_demand
                        repairs += 1
        
        return repairs
    
    def _get_nearest_warehouse(self, customer_k: int) -> int:
        """Get nearest warehouse to customer."""
        distances = self.network.distance_wc[:, customer_k]
        return int(np.argmin(distances))
    
    def _get_nearest_among(self, warehouse_list: List[int], customer_k: int) -> int:
        """Get nearest warehouse from a list."""
        distances = {
            j: self.network.distance_wc[j, customer_k]
            for j in warehouse_list
        }
        return min(distances, key=distances.get)
    
    def _get_nearest_supplier(self, warehouse_j: int) -> int:
        """Get nearest supplier to warehouse."""
        distances = self.network.distance_sw[:, warehouse_j]
        return int(np.argmin(distances))
    
    def _find_alternative_warehouse(
        self,
        customer_k: int,
        exclude_j: int,
        y_assign: Dict[Tuple[int, int], int]
    ) -> int:
        """Find alternative warehouse with capacity."""
        num_w = self.network.num_warehouses
        
        # Get warehouses sorted by distance
        distances = [
            (j, self.network.distance_wc[j, customer_k])
            for j in range(num_w)
            if j != exclude_j
        ]
        distances.sort(key=lambda x: x[1])
        
        # Find first warehouse with capacity
        for j, _ in distances:
            warehouse = self.network.get_warehouse(j)
            
            # Calculate current load
            current_load = sum(
                self.network.get_customer(k).average_demand
                for k in range(self.network.num_customers)
                if y_assign.get((j, k), 0) == 1
            )
            
            # Check if can accommodate
            new_demand = self.network.get_customer(customer_k).average_demand
            if current_load + new_demand <= warehouse.capacity * 0.9:
                return j
        
        return None
    
    def compute_feasibility_metrics(
        self,
        x_assign: Dict[Tuple[int, int], int],
        y_assign: Dict[Tuple[int, int], int]
    ) -> Dict[str, float]:
        """
        Compute feasibility metrics for current assignments.
        
        Returns:
            Dict with constraint violation counts
        """
        num_w = self.network.num_warehouses
        num_c = self.network.num_customers
        num_s = self.network.num_suppliers
        
        metrics = {
            "vendor_violations": 0,
            "factory_violations": 0,
            "connectivity_violations": 0,
            "total_violations": 0
        }
        
        # Check vendor single-sourcing
        for k in range(1, num_c):
            count = sum(y_assign.get((j, k), 0) for j in range(num_w))
            if count != 1:
                metrics["vendor_violations"] += 1
        
        # Check factory redundancy
        factory_count = sum(y_assign.get((j, 0), 0) for j in range(num_w))
        if factory_count != 2:
            metrics["factory_violations"] = 1
        
        # Check supplier connectivity
        for j in range(num_w):
            has_customers = any(y_assign.get((j, k), 0) == 1 for k in range(num_c))
            has_suppliers = any(x_assign.get((i, j), 0) == 1 for i in range(num_s))
            if has_customers and not has_suppliers:
                metrics["connectivity_violations"] += 1
        
        metrics["total_violations"] = (
            metrics["vendor_violations"] +
            metrics["factory_violations"] +
            metrics["connectivity_violations"]
        )
        
        return metrics
