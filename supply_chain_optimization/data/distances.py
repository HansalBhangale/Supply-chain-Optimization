"""
Distance matrix computation utilities.

Computes D^SW (supplier-warehouse) and D^WC (warehouse-customer) distances.
"""

from typing import Tuple
import numpy as np

from .models import SupplyChainNetwork


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_distance_matrices(network: SupplyChainNetwork) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distance matrices for the supply chain network.
    
    Mathematical notation:
        D^SW_{i,j}: supplier i → warehouse j distance
        D^WC_{j,k}: warehouse j → customer k distance
    
    Args:
        network: The supply chain network
        
    Returns:
        Tuple of (D_SW, D_WC) distance matrices
    """
    num_s = network.num_suppliers
    num_w = network.num_warehouses
    num_c = network.num_customers
    
    # D^SW: Supplier to Warehouse distances (|S| x |W|)
    d_sw = np.zeros((num_s, num_w))
    for i, supplier in enumerate(network.suppliers):
        for j, warehouse in enumerate(network.warehouses):
            d_sw[i, j] = euclidean_distance(supplier.location, warehouse.location)
    
    # D^WC: Warehouse to Customer distances (|W| x |C|)
    d_wc = np.zeros((num_w, num_c))
    for j, warehouse in enumerate(network.warehouses):
        for k, customer in enumerate(network.customers):
            d_wc[j, k] = euclidean_distance(warehouse.location, customer.location)
    
    # Store in network object
    network.distance_sw = d_sw
    network.distance_wc = d_wc
    
    return d_sw, d_wc


def compute_lead_time_matrices(
    network: SupplyChainNetwork,
    sw_base: int = 2,
    wc_base: int = 1,
    distance_factor: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lead time matrices based on distances.
    
    Mathematical notation:
        L^SW_{i,j}: supplier i → warehouse j lead time (days)
        L^WC_{j,k}: warehouse j → customer k lead time (days)
    
    Args:
        network: The supply chain network
        sw_base: Base lead time for supplier-warehouse
        wc_base: Base lead time for warehouse-customer
        distance_factor: Factor to add lead time based on distance
        
    Returns:
        Tuple of (L_SW, L_WC) lead time matrices (integer days)
    """
    # Ensure distance matrices are computed
    if network.distance_sw is None or network.distance_wc is None:
        compute_distance_matrices(network)
    
    # Lead times: base + distance-dependent component
    l_sw = np.ceil(sw_base + network.distance_sw * distance_factor).astype(int)
    l_wc = np.ceil(wc_base + network.distance_wc * distance_factor).astype(int)
    
    # Store in network object
    network.lead_time_sw = l_sw
    network.lead_time_wc = l_wc
    
    return l_sw, l_wc


def get_nearest_warehouse(network: SupplyChainNetwork, customer_idx: int) -> int:
    """
    Find the nearest warehouse to a customer.
    
    Args:
        network: The supply chain network
        customer_idx: Index of the customer
        
    Returns:
        Index of the nearest warehouse
    """
    if network.distance_wc is None:
        compute_distance_matrices(network)
    
    return int(np.argmin(network.distance_wc[:, customer_idx]))


def get_nearest_supplier(network: SupplyChainNetwork, warehouse_idx: int) -> int:
    """
    Find the nearest supplier to a warehouse.
    
    Args:
        network: The supply chain network
        warehouse_idx: Index of the warehouse
        
    Returns:
        Index of the nearest supplier
    """
    if network.distance_sw is None:
        compute_distance_matrices(network)
    
    return int(np.argmin(network.distance_sw[:, warehouse_idx]))
