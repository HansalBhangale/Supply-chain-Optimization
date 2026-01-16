"""
Scalability Benchmark Suite for Supply Chain Optimization.

Tests the optimization across different network sizes to measure:
- Execution time
- Memory usage
- Solution quality
- Feasibility rate
"""

import time
import json
import tracemalloc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from supply_chain_optimization.config import SupplyChainConfig, NetworkConfig, ScenarioConfig, QAOAConfig
from supply_chain_optimization.data.models import create_sample_network
from supply_chain_optimization.data.distances import compute_distance_matrices, compute_lead_time_matrices
from supply_chain_optimization.data.scenarios import generate_demand_scenarios
from supply_chain_optimization.hybrid.coordinator import HybridSolver


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    num_suppliers: int
    num_warehouses: int
    num_customers: int
    
    @property
    def qubo_vars(self) -> int:
        """Estimated QUBO variables: S*W + W*C"""
        return self.num_suppliers * self.num_warehouses + \
               self.num_warehouses * self.num_customers


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    solver_type: str
    qubo_variables: int
    solve_time_seconds: float
    memory_peak_mb: float
    objective_value: float
    is_feasible: bool
    num_supplier_routes: int
    num_customer_routes: int


# Benchmark configurations
BENCHMARK_CONFIGS = [
    BenchmarkConfig("Tiny", 3, 2, 5),
    BenchmarkConfig("Small", 5, 3, 10),
    BenchmarkConfig("Medium", 10, 5, 20),
    BenchmarkConfig("Large", 20, 10, 50),
]


def run_single_benchmark(
    bench_config: BenchmarkConfig,
    solver_type: str = "classical",
    random_seed: int = 42
) -> BenchmarkResult:
    """
    Run a single benchmark configuration.
    
    Args:
        bench_config: Benchmark configuration
        solver_type: 'classical', 'qaoa', or 'ibm_quantum'
        random_seed: Random seed for reproducibility
        
    Returns:
        BenchmarkResult with metrics
    """
    print(f"\n{'='*60}")
    print(f"Running: {bench_config.name} ({bench_config.qubo_vars} QUBO vars) - {solver_type}")
    print(f"{'='*60}")
    
    # Create config
    config = SupplyChainConfig(
        network=NetworkConfig(
            num_suppliers=bench_config.num_suppliers,
            num_warehouses=bench_config.num_warehouses,
            num_customers=bench_config.num_customers,
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
        random_seed=random_seed
    )
    
    # Create network
    network = create_sample_network(
        num_suppliers=config.network.num_suppliers,
        num_warehouses=config.network.num_warehouses,
        num_customers=config.network.num_customers,
        random_seed=random_seed
    )
    compute_distance_matrices(network)
    compute_lead_time_matrices(network)
    
    # Generate scenarios
    scenarios = generate_demand_scenarios(network, config.scenarios, random_seed)
    
    # Start tracking memory
    tracemalloc.start()
    
    # Run optimization with timing
    start_time = time.time()
    
    try:
        solver = HybridSolver(network, config, scenarios)
        
        if solver_type == "classical":
            result = solver.solve(use_classical_stage1=True)
        elif solver_type == "qaoa":
            result = solver.solve(use_classical_stage1=False)
        elif solver_type == "ibm_quantum":
            # Use IBM Quantum backend
            result = solver.solve(use_classical_stage1=False, use_ibm_quantum=True)
        else:
            result = solver.solve(use_classical_stage1=True)
        
        solve_time = time.time() - start_time
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Check feasibility
        is_feasible = result.is_feasible if hasattr(result, 'is_feasible') else True
        
        # Count routes
        num_sw = sum(1 for v in result.x_assign.values() if v == 1)
        num_wc = sum(1 for v in result.y_assign.values() if v == 1)
        
        print(f"  Time: {solve_time:.2f}s | Memory: {peak/1e6:.1f}MB | Objective: {result.total_objective:.0f}")
        
        return BenchmarkResult(
            config_name=bench_config.name,
            solver_type=solver_type,
            qubo_variables=bench_config.qubo_vars,
            solve_time_seconds=solve_time,
            memory_peak_mb=peak / 1e6,
            objective_value=result.total_objective,
            is_feasible=is_feasible,
            num_supplier_routes=num_sw,
            num_customer_routes=num_wc
        )
        
    except Exception as e:
        tracemalloc.stop()
        print(f"  FAILED: {type(e).__name__}: {e}")
        return BenchmarkResult(
            config_name=bench_config.name,
            solver_type=solver_type,
            qubo_variables=bench_config.qubo_vars,
            solve_time_seconds=time.time() - start_time,
            memory_peak_mb=0,
            objective_value=float('inf'),
            is_feasible=False,
            num_supplier_routes=0,
            num_customer_routes=0
        )


def run_all_benchmarks(
    configs: List[BenchmarkConfig] = None,
    solver_types: List[str] = None,
    output_dir: str = "results"
) -> List[BenchmarkResult]:
    """
    Run all benchmark configurations.
    
    Args:
        configs: List of benchmark configs (default: BENCHMARK_CONFIGS)
        solver_types: List of solver types to test
        output_dir: Directory to save results
        
    Returns:
        List of BenchmarkResult
    """
    if configs is None:
        configs = BENCHMARK_CONFIGS
    
    if solver_types is None:
        solver_types = ["classical"]
    
    results = []
    
    print("\n" + "="*60)
    print("SUPPLY CHAIN OPTIMIZATION - SCALABILITY BENCHMARK")
    print("="*60)
    
    for config in configs:
        for solver in solver_types:
            try:
                result = run_single_benchmark(config, solver)
                results.append(result)
            except Exception as e:
                print(f"  Error in {config.name}/{solver}: {e}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / "scalability_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Config':<10} {'Solver':<12} {'QUBO Vars':<10} {'Time (s)':<10} {'Mem (MB)':<10} {'Objective':<12}")
    print("-"*80)
    
    for r in results:
        obj_str = f"{r.objective_value:.0f}" if r.objective_value < 1e10 else "INFEASIBLE"
        print(f"{r.config_name:<10} {r.solver_type:<12} {r.qubo_variables:<10} {r.solve_time_seconds:<10.2f} {r.memory_peak_mb:<10.1f} {obj_str:<12}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run scalability benchmarks")
    parser.add_argument("--solvers", nargs="+", default=["classical"], 
                        choices=["classical", "qaoa", "ibm_quantum"],
                        help="Solver types to benchmark")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config names to run (Tiny, Small, Medium, Large)")
    parser.add_argument("--output", default="results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Filter configs if specified
    configs = BENCHMARK_CONFIGS
    if args.configs:
        configs = [c for c in BENCHMARK_CONFIGS if c.name in args.configs]
    
    run_all_benchmarks(configs, args.solvers, args.output)
