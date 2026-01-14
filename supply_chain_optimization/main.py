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
    
    # Create sample network
    print("\n[Building Network]")
    network = create_sample_network(
        num_suppliers=config.network.num_suppliers,
        num_warehouses=config.network.num_warehouses,
        num_customers=config.network.num_customers,
        random_seed=config.random_seed
    )
    
    # Compute matrices
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
