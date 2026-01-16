"""
IBM Quantum Hardware Test

Runs a tiny supply chain optimization problem on real IBM Quantum hardware.
This test uses the smallest possible problem to:
1. Verify IBM Quantum connectivity
2. Run QAOA on actual quantum hardware
3. Compare results with simulator
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env from project root and supply_chain_optimization folder
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
load_dotenv(project_root / "supply_chain_optimization" / ".env")

# Check for IBM Quantum token
token = os.getenv("IBM_QUANTUM_TOKEN") or os.getenv("IBMQ_TOKEN")
if not token:
    print("ERROR: IBM_QUANTUM_TOKEN not found in .env file")
    print("Please add: IBM_QUANTUM_TOKEN=your_token_here")
    sys.exit(1)

print("="*60)
print("IBM QUANTUM HARDWARE TEST")
print("="*60)
print(f"Token found: {'*' * 10}...{token[-4:]}")

from supply_chain_optimization.config import SupplyChainConfig, NetworkConfig, ScenarioConfig, QAOAConfig
from supply_chain_optimization.data.models import create_sample_network
from supply_chain_optimization.data.distances import compute_distance_matrices, compute_lead_time_matrices
from supply_chain_optimization.data.scenarios import generate_demand_scenarios
from supply_chain_optimization.stage1_qubo.qubo_builder import QUBOBuilder
from supply_chain_optimization.stage1_qubo.qaoa_solver import QAOASolver


def run_ibm_quantum_test():
    """Run a tiny problem on IBM Quantum hardware."""
    
    # Create TINY config (2 suppliers, 2 warehouses, 3 customers = ~12 QUBO vars)
    print("\n[1] Creating tiny test problem...")
    config = SupplyChainConfig(
        network=NetworkConfig(
            num_suppliers=2,
            num_warehouses=2,
            num_customers=3,
        ),
        scenarios=ScenarioConfig(num_scenarios=1, horizon_days=3),
        qaoa=QAOAConfig(depth=1, shots=512, maxiter=30),
        random_seed=42
    )
    
    # Create network
    network = create_sample_network(
        num_suppliers=2,
        num_warehouses=2,
        num_customers=3,
        random_seed=42
    )
    compute_distance_matrices(network)
    compute_lead_time_matrices(network)
    
    # Build QUBO
    print("\n[2] Building QUBO matrix...")
    qubo_builder = QUBOBuilder(network, config)
    Q, constant, indexer = qubo_builder.build()
    
    n_vars = Q.shape[0]
    print(f"  QUBO size: {n_vars} variables (qubits)")
    
    if n_vars > 25:
        print(f"  WARNING: Problem size {n_vars} > 25, may be too large for free tier")
    
    # Create solver
    solver = QAOASolver(config.qaoa)
    
    # Run on SIMULATOR first
    print("\n[3] Running on LOCAL SIMULATOR...")
    sim_start = time.time()
    try:
        sim_bitstring, sim_value = solver.solve(Q, constant, indexer)
        sim_time = time.time() - sim_start
        print(f"  Simulator result: {sim_value:.2f}")
        print(f"  Simulator time: {sim_time:.2f}s")
    except Exception as e:
        print(f"  Simulator failed: {e}")
        sim_value = None
        sim_time = 0
    
    # Run on IBM QUANTUM
    print("\n[4] Running on IBM QUANTUM HARDWARE...")
    print("  ⏳ This may take several minutes (queue + execution)")
    print("  Connecting to IBM Quantum...")
    
    ibm_start = time.time()
    try:
        ibm_bitstring, ibm_value = solver.solve_ibm_quantum(Q, constant)
        ibm_time = time.time() - ibm_start
        print(f"\n  ✅ IBM Quantum result: {ibm_value:.2f}")
        print(f"  IBM Quantum time: {ibm_time:.2f}s")
    except Exception as e:
        print(f"\n  ❌ IBM Quantum failed: {type(e).__name__}: {e}")
        ibm_value = None
        ibm_time = 0
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Method':<20} {'Objective':>15} {'Time (s)':>10}")
    print("-"*45)
    
    if sim_value is not None:
        print(f"{'Simulator':<20} {sim_value:>15.2f} {sim_time:>10.2f}")
    
    if ibm_value is not None:
        print(f"{'IBM Quantum':<20} {ibm_value:>15.2f} {ibm_time:>10.2f}")
        
        if sim_value is not None:
            diff = abs(ibm_value - sim_value) / abs(sim_value) * 100
            print(f"\nDifference: {diff:.1f}%")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_ibm_quantum_test()
