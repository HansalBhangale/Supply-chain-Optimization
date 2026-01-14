"""
Main entry point for supply chain optimization.

Run with: uv run python -m supply_chain_optimization.main
"""

import argparse
import sys
from pathlib import Path

from .config import SupplyChainConfig, create_default_config, create_small_config
from .data.models import create_sample_network
from .data.distances import compute_distance_matrices, compute_lead_time_matrices
from .data.scenarios import generate_demand_scenarios
from .hybrid.coordinator import HybridSolver
from .utils.visualization import plot_results


def print_route_decisions(network, result):
    """Print detailed route decisions in a customer-friendly format."""
    print("\n" + "=" * 60)
    print("OPTIMIZED SUPPLY CHAIN DECISIONS")
    print("=" * 60)
    
    # Supplier -> Warehouse assignments
    print("\n[*] SUPPLIER -> WAREHOUSE ROUTES (Active)")
    print("-" * 50)
    
    active_sw = []
    for (i, j), val in result.x_assign.items():
        if val == 1:
            supplier = network.suppliers[i]
            warehouse = network.warehouses[j]
            distance = network.distance_sw[i, j] if network.distance_sw is not None else 0
            lead_time = network.lead_time_sw[i, j] if network.lead_time_sw is not None else 0
            active_sw.append((supplier.name, warehouse.name, distance, lead_time))
    
    if active_sw:
        print(f"{'Supplier':<25} {'Warehouse':<20} {'Distance (km)':<15} {'Lead Time (days)'}")
        print("-" * 80)
        for supplier, warehouse, dist, lt in sorted(active_sw):
            print(f"{supplier:<25} {warehouse:<20} {dist:>10.1f} km    {lt:>5.0f} days")
    else:
        print("  No active supplier routes found.")
    
    # Warehouse -> Customer assignments
    print("\n[*] WAREHOUSE -> CUSTOMER ROUTES (Active)")
    print("-" * 50)
    
    # Group by warehouse
    warehouse_customers = {}
    for (j, k), val in result.y_assign.items():
        if val == 1:
            warehouse = network.warehouses[j]
            customer = network.customers[k]
            distance = network.distance_wc[j, k] if network.distance_wc is not None else 0
            lead_time = network.lead_time_wc[j, k] if network.lead_time_wc is not None else 0
            
            if warehouse.name not in warehouse_customers:
                warehouse_customers[warehouse.name] = []
            
            customer_type = "[FACTORY]" if customer.is_factory else f"Tier {customer.tier}"
            warehouse_customers[warehouse.name].append(
                (customer.name, customer_type, distance, lead_time)
            )
    
    for warehouse_name in sorted(warehouse_customers.keys()):
        customers = warehouse_customers[warehouse_name]
        print(f"\n  [>] {warehouse_name} serves {len(customers)} customers:")
        print(f"     {'Customer':<25} {'Type':<12} {'Distance':<12} {'Lead Time'}")
        print("     " + "-" * 65)
        for cust_name, cust_type, dist, lt in sorted(customers, key=lambda x: x[2]):
            print(f"     {cust_name:<25} {cust_type:<12} {dist:>8.1f} km   {lt:>5.0f} days")
    
    # Summary recommendations
    print("\n" + "=" * 60)
    print("[+] RECOMMENDED ACTIONS")
    print("=" * 60)
    
    total_sw = len(active_sw)
    total_wc = sum(len(v) for v in warehouse_customers.values())
    
    print(f"""
1. ACTIVATE {total_sw} supplier contracts:
   Connect the above suppliers to their assigned warehouses.

2. SETUP {total_wc} distribution routes:
   Assign delivery trucks from each warehouse to its customers.

3. MAINTAIN INVENTORY based on lead times:
   - Suppliers take ~{sum(x[3] for x in active_sw)/max(1,len(active_sw)):.1f} days avg to reach warehouses
   - Customers receive within ~{sum(c[3] for v in warehouse_customers.values() for c in v)/max(1,total_wc):.1f} days from warehouse

4. FACTORY REDUNDANCY ensured:
   Factory is served by multiple warehouses for reliability.
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum-enabled Supply Chain Optimization"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run with small test configuration"
    )
    parser.add_argument(
        "--classical",
        action="store_true",
        help="Use classical solver instead of QAOA for Stage 1"
    )
    parser.add_argument(
        "--suppliers",
        type=int,
        default=10,
        help="Number of suppliers"
    )
    parser.add_argument(
        "--warehouses",
        type=int,
        default=5,
        help="Number of warehouses"
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=20,
        help="Number of customers (including factory)"
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=3,
        help="Number of demand scenarios"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=28,
        help="Planning horizon in days"
    )
    parser.add_argument(
        "--qaoa-depth",
        type=int,
        default=2,
        help="QAOA circuit depth"
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        help="Directory to save plots"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use-ors",
        action="store_true",
        help="Use OpenRouteService API for real distances (FREE, 2000 requests/day)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("QUANTUM SUPPLY CHAIN OPTIMIZATION")
    print("Two-Stage Stochastic Programming with QAOA + MILP")
    print("=" * 60)
    
    # Create configuration
    if args.test_mode:
        config = create_small_config()
        print("\n[Config] Using small test configuration")
    else:
        config = create_default_config()
        config.network.num_suppliers = args.suppliers
        config.network.num_warehouses = args.warehouses
        config.network.num_customers = args.customers
        config.scenarios.num_scenarios = args.scenarios
        config.scenarios.horizon_days = args.horizon
        config.qaoa.depth = args.qaoa_depth
        config.random_seed = args.seed
    
    print(f"\n[Network Configuration]")
    print(f"  Suppliers: {config.network.num_suppliers}")
    print(f"  Warehouses: {config.network.num_warehouses}")
    print(f"  Customers: {config.network.num_customers}")
    print(f"  Scenarios: {config.scenarios.num_scenarios}")
    print(f"  Horizon: {config.scenarios.horizon_days} days")
    print(f"  QAOA depth: {config.qaoa.depth}")
    
    distance_mode = "OpenRouteService (FREE)" if args.use_ors else "Euclidean"
    print(f"  Distance mode: {distance_mode}")
    
    # Create sample network
    print("\n[Building Network]")
    
    if args.use_ors:
        # Use OpenRouteService (FREE, 2000 requests/day)
        from .data.openrouteservice_client import SAMPLE_INDIAN_CITIES
        from .data.openrouteservice_client import compute_ors_distance_matrices
        from .data.models import Supplier, Warehouse, Customer, SupplyChainNetwork
        import numpy as np
        
        print("  Using OpenRouteService API (FREE, open-source)")
        
        # Get sample cities
        supplier_locs = SAMPLE_INDIAN_CITIES["suppliers"][:config.network.num_suppliers]
        warehouse_locs = SAMPLE_INDIAN_CITIES["warehouses"][:config.network.num_warehouses]
        customer_locs = SAMPLE_INDIAN_CITIES["customers"][:config.network.num_customers]
        
        # Create network entities
        suppliers = [
            Supplier(
                id=i, name=loc.name, 
                location=(loc.lat, loc.lng),
                capacity={t: np.random.uniform(800, 1200) for t in range(config.scenarios.horizon_days)}
            )
            for i, loc in enumerate(supplier_locs)
        ]
        
        warehouses = [
            Warehouse(
                id=j, name=loc.name,
                location=(loc.lat, loc.lng),
                capacity=np.random.uniform(4000, 6000),
                holding_cost=np.random.uniform(0.8, 1.2),
                initial_inventory=np.random.uniform(500, 1500)
            )
            for j, loc in enumerate(warehouse_locs)
        ]
        
        customers = []
        for k, loc in enumerate(customer_locs):
            is_factory = (k == 0)
            tier = 0 if is_factory else (1 if k <= len(customer_locs)//4 else (2 if k <= len(customer_locs)//2 else 3))
            customers.append(Customer(
                id=k, name=loc.name,
                location=(loc.lat, loc.lng),
                is_factory=is_factory,
                shortage_penalty=10000.0 if is_factory else {1: 100.0, 2: 50.0, 3: 25.0}.get(tier, 50.0),
                service_level_z=3.0 if is_factory else {1: 2.33, 2: 1.96, 3: 1.64}.get(tier, 1.96),
                tier=tier,
                average_demand=200.0 if is_factory else np.random.uniform(30, 70)
            ))
        
        network = SupplyChainNetwork(
            suppliers=suppliers,
            warehouses=warehouses,
            customers=customers
        )
        
        # Compute real distances using ORS
        supplier_coords = [(loc.lat, loc.lng) for loc in supplier_locs]
        warehouse_coords = [(loc.lat, loc.lng) for loc in warehouse_locs]
        customer_coords = [(loc.lat, loc.lng) for loc in customer_locs]
        
        compute_ors_distance_matrices(network, supplier_coords, warehouse_coords, customer_coords)
        
    else:
        # Use synthetic data with Euclidean distances
        network = create_sample_network(
            num_suppliers=config.network.num_suppliers,
            num_warehouses=config.network.num_warehouses,
            num_customers=config.network.num_customers,
            random_seed=config.random_seed
        )
        compute_distance_matrices(network)
        compute_lead_time_matrices(network)
    
    print(f"  Network created with {network.num_suppliers} suppliers, "
          f"{network.num_warehouses} warehouses, {network.num_customers} customers")
    
    # Generate scenarios
    print("\n[Generating Demand Scenarios]")
    scenarios = generate_demand_scenarios(
        network,
        config.scenarios,
        random_seed=config.random_seed
    )
    
    for s in scenarios:
        total_demand = s.demand.sum()
        print(f"  Scenario {s.id}: prob={s.probability:.2f}, total_demand={total_demand:.0f}")
    
    # Run hybrid solver
    solver = HybridSolver(network, config, scenarios)
    
    result = solver.solve(
        use_classical_stage1=args.classical,
        milp_solver='PULP_CBC_CMD',
        milp_time_limit=300
    )
    
    # Print network summary
    print(f"\n[Network Assignment Summary]")
    summary = solver.get_network_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Print detailed route decisions
    print_route_decisions(network, result)
    
    # Visualize results
    if args.save_plots:
        print(f"\n[Saving plots to {args.save_plots}]")
        Path(args.save_plots).mkdir(parents=True, exist_ok=True)
        plot_results(network, result, save_dir=args.save_plots)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
