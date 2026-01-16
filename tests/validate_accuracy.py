"""
Accuracy Validation Suite for Supply Chain Optimization.

Compares optimization solutions against:
1. Brute-force enumeration (for tiny instances)
2. Multiple random seeds (consistency analysis)
3. Known constraints (feasibility verification)
"""

import time
import json
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from supply_chain_optimization.config import SupplyChainConfig, NetworkConfig, ScenarioConfig, QAOAConfig
from supply_chain_optimization.data.models import create_sample_network, SupplyChainNetwork
from supply_chain_optimization.data.distances import compute_distance_matrices, compute_lead_time_matrices
from supply_chain_optimization.data.scenarios import generate_demand_scenarios
from supply_chain_optimization.stage1_qubo.qubo_builder import QUBOBuilder
from supply_chain_optimization.hybrid.coordinator import HybridSolver


@dataclass
class ValidationResult:
    """Results from accuracy validation."""
    test_name: str
    solver_type: str
    solver_objective: float
    brute_force_objective: Optional[float]
    optimality_gap_percent: Optional[float]
    is_feasible: bool
    constraint_violations: Dict[str, int]
    solve_time_seconds: float


def compute_qubo_energy(bitstring: np.ndarray, Q: np.ndarray, constant: float = 0.0) -> float:
    """Compute QUBO energy: E(z) = z^T Q z + constant"""
    return float(bitstring @ Q @ bitstring + constant)


def brute_force_solve(Q: np.ndarray, constant: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Find optimal QUBO solution by exhaustive enumeration.
    WARNING: Exponential complexity - only use for n <= 15
    
    Returns:
        Tuple of (optimal_bitstring, optimal_value)
    """
    n = Q.shape[0]
    
    if n > 15:
        raise ValueError(f"Brute force not feasible for n={n} (2^{n} = {2**n} combinations)")
    
    best_bitstring = None
    best_value = float('inf')
    
    total = 2 ** n
    print(f"  Checking {total:,} combinations...")
    
    for i, bits in enumerate(itertools.product([0, 1], repeat=n)):
        bitstring = np.array(bits)
        value = compute_qubo_energy(bitstring, Q, constant)
        
        if value < best_value:
            best_value = value
            best_bitstring = bitstring.copy()
        
        if i % 10000 == 0 and i > 0:
            print(f"    Progress: {i/total*100:.1f}%")
    
    return best_bitstring, best_value


def check_constraints(
    x_assign: Dict,
    y_assign: Dict,
    network: SupplyChainNetwork
) -> Dict[str, int]:
    """
    Check constraint satisfaction.
    
    Returns:
        Dict of constraint name -> violation count
    """
    violations = {}
    
    # 1. Single-sourcing: each non-factory customer has exactly one warehouse
    violations["single_sourcing"] = 0
    for k in range(1, len(network.customers)):  # Skip factory (k=0)
        serving_warehouses = sum(1 for j in range(len(network.warehouses)) 
                                  if y_assign.get((j, k), 0) == 1)
        if serving_warehouses != 1:
            violations["single_sourcing"] += 1
    
    # 2. Factory redundancy: factory must have >= 2 warehouses
    factory_warehouses = sum(1 for j in range(len(network.warehouses))
                              if y_assign.get((j, 0), 0) == 1)
    violations["factory_redundancy"] = 0 if factory_warehouses >= 2 else 1
    
    # 3. Supplier connectivity: active warehouse has at least one supplier
    violations["supplier_connectivity"] = 0
    for j in range(len(network.warehouses)):
        serves_anyone = any(y_assign.get((j, k), 0) == 1 
                           for k in range(len(network.customers)))
        if serves_anyone:
            has_supplier = any(x_assign.get((i, j), 0) == 1 
                              for i in range(len(network.suppliers)))
            if not has_supplier:
                violations["supplier_connectivity"] += 1
    
    return violations


def run_validation_test(
    num_suppliers: int = 3,
    num_warehouses: int = 2,
    num_customers: int = 5,
    solver_type: str = "classical",
    random_seed: int = 42
) -> ValidationResult:
    """
    Run a single validation test comparing solver vs brute-force.
    """
    test_name = f"{num_suppliers}S_{num_warehouses}W_{num_customers}C"
    print(f"\n{'='*60}")
    print(f"Validation Test: {test_name} - {solver_type}")
    print(f"{'='*60}")
    
    # Create config
    config = SupplyChainConfig(
        network=NetworkConfig(
            num_suppliers=num_suppliers,
            num_warehouses=num_warehouses,
            num_customers=num_customers,
        ),
        scenarios=ScenarioConfig(num_scenarios=2, horizon_days=7),
        qaoa=QAOAConfig(depth=1, shots=512, maxiter=50),
        random_seed=random_seed
    )
    
    # Create network
    network = create_sample_network(
        num_suppliers=num_suppliers,
        num_warehouses=num_warehouses,
        num_customers=num_customers,
        random_seed=random_seed
    )
    compute_distance_matrices(network)
    compute_lead_time_matrices(network)
    
    # Generate scenarios
    scenarios = generate_demand_scenarios(network, config.scenarios, random_seed)
    
    # Build QUBO for brute-force comparison
    qubo_builder = QUBOBuilder(network, config)
    Q, constant, indexer = qubo_builder.build()
    n_vars = Q.shape[0]
    
    # Run solver
    start_time = time.time()
    solver = HybridSolver(network, config, scenarios)
    result = solver.solve(use_classical_stage1=(solver_type == "classical"))
    solve_time = time.time() - start_time
    
    # Check constraints
    violations = check_constraints(result.x_assign, result.y_assign, network)
    is_feasible = all(v == 0 for v in violations.values())
    
    # Brute-force comparison (only for small problems)
    brute_force_obj = None
    optimality_gap = None
    
    if n_vars <= 15:
        print(f"  Running brute-force enumeration ({n_vars} variables)...")
        bf_bitstring, bf_value = brute_force_solve(Q, constant)
        brute_force_obj = bf_value
        
        # Calculate optimality gap
        solver_stage1 = result.stage1_value
        if bf_value != 0:
            optimality_gap = abs(solver_stage1 - bf_value) / abs(bf_value) * 100
        else:
            optimality_gap = 0.0 if solver_stage1 == 0 else float('inf')
        
        print(f"  Brute-force optimal: {bf_value:.2f}")
        print(f"  Solver Stage-1 value: {solver_stage1:.2f}")
        print(f"  Optimality gap: {optimality_gap:.2f}%")
    else:
        print(f"  Skipping brute-force (n={n_vars} > 15)")
    
    print(f"  Feasible: {is_feasible}")
    print(f"  Constraint violations: {violations}")
    
    return ValidationResult(
        test_name=test_name,
        solver_type=solver_type,
        solver_objective=result.total_objective,
        brute_force_objective=brute_force_obj,
        optimality_gap_percent=optimality_gap,
        is_feasible=is_feasible,
        constraint_violations=violations,
        solve_time_seconds=solve_time
    )


def run_consistency_analysis(
    num_suppliers: int = 5,
    num_warehouses: int = 3,
    num_customers: int = 10,
    num_runs: int = 5,
    solver_type: str = "classical"
) -> Dict[str, Any]:
    """
    Run multiple optimization runs with different seeds to check consistency.
    """
    print(f"\n{'='*60}")
    print(f"Consistency Analysis: {num_runs} runs")
    print(f"{'='*60}")
    
    objectives = []
    solve_times = []
    
    for seed in range(num_runs):
        config = SupplyChainConfig(
            network=NetworkConfig(
                num_suppliers=num_suppliers,
                num_warehouses=num_warehouses,
                num_customers=num_customers,
            ),
            scenarios=ScenarioConfig(num_scenarios=2, horizon_days=7),
            random_seed=seed
        )
        
        network = create_sample_network(
            num_suppliers=num_suppliers,
            num_warehouses=num_warehouses,
            num_customers=num_customers,
            random_seed=seed
        )
        compute_distance_matrices(network)
        compute_lead_time_matrices(network)
        
        scenarios = generate_demand_scenarios(network, config.scenarios, seed)
        
        start_time = time.time()
        solver = HybridSolver(network, config, scenarios)
        result = solver.solve(use_classical_stage1=(solver_type == "classical"))
        solve_time = time.time() - start_time
        
        objectives.append(result.total_objective)
        solve_times.append(solve_time)
        
        print(f"  Run {seed+1}: Objective={result.total_objective:.0f}, Time={solve_time:.2f}s")
    
    # Statistics
    obj_mean = np.mean(objectives)
    obj_std = np.std(objectives)
    obj_cv = obj_std / obj_mean * 100 if obj_mean != 0 else 0
    
    print(f"\n  Mean objective: {obj_mean:.0f}")
    print(f"  Std deviation: {obj_std:.0f}")
    print(f"  Coefficient of variation: {obj_cv:.1f}%")
    
    return {
        "solver_type": solver_type,
        "num_runs": num_runs,
        "objectives": objectives,
        "mean_objective": obj_mean,
        "std_objective": obj_std,
        "cv_percent": obj_cv,
        "mean_solve_time": np.mean(solve_times)
    }


def run_full_validation(output_dir: str = "results") -> Dict[str, Any]:
    """
    Run complete validation suite.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("SUPPLY CHAIN OPTIMIZATION - ACCURACY VALIDATION")
    print("="*60)
    
    results = {
        "accuracy_tests": [],
        "consistency_analysis": None
    }
    
    # Run accuracy tests for tiny instances
    for solver in ["classical"]:
        for config in [(3, 2, 5), (3, 2, 4), (2, 2, 4)]:
            result = run_validation_test(
                num_suppliers=config[0],
                num_warehouses=config[1],
                num_customers=config[2],
                solver_type=solver
            )
            results["accuracy_tests"].append(asdict(result))
    
    # Run consistency analysis
    results["consistency_analysis"] = run_consistency_analysis(
        num_suppliers=5,
        num_warehouses=3,
        num_customers=10,
        num_runs=5,
        solver_type="classical"
    )
    
    # Save results
    results_file = output_path / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for test in results["accuracy_tests"]:
        gap_str = f"{test['optimality_gap_percent']:.1f}%" if test['optimality_gap_percent'] is not None else "N/A"
        feasible_str = "✓" if test['is_feasible'] else "✗"
        print(f"{test['test_name']:<15} Gap: {gap_str:<8} Feasible: {feasible_str}")
    
    if results["consistency_analysis"]:
        cv = results["consistency_analysis"]["cv_percent"]
        print(f"\nConsistency (CV): {cv:.1f}% {'(GOOD)' if cv < 10 else '(HIGH VARIANCE)'}")
    
    return results


if __name__ == "__main__":
    run_full_validation()
