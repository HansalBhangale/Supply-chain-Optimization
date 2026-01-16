"""
QAOA solver integration with IBM Qiskit.

Solves Stage 1 QUBO using Quantum Approximate Optimization Algorithm.
Supports both local simulator and real IBM Quantum hardware.
"""

import os
from typing import Dict, Tuple, Optional, List
import numpy as np

# Qiskit imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer, 
    GroverOptimizer,
    GurobiOptimizer,
)
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP

# IBM Quantum Runtime (optional)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
    IBM_QUANTUM_AVAILABLE = True
except ImportError:
    IBM_QUANTUM_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

from .qubo_builder import QUBOBuilder, VariableIndexer
from ..config import QAOAConfig


class QAOASolver:
    """
    QAOA solver for Stage 1 strategic assignment.
    
    Uses Qiskit's QAOA implementation with local Aer simulator.
    For larger problems, falls back to simulated annealing or greedy heuristics.
    """
    
    def __init__(self, config: QAOAConfig):
        """
        Initialize QAOA solver.
        
        Args:
            config: QAOA configuration (depth, shots, optimizer, etc.)
        """
        self.config = config
        self.depth = config.depth
        self.shots = config.shots
        self.maxiter = config.maxiter
        
        # Get classical optimizer
        self.optimizer = self._get_optimizer(config.optimizer)
        
        # Store results
        self.result = None
        self.optimal_bitstring: Optional[np.ndarray] = None
        self.optimal_value: Optional[float] = None
    
    def _get_optimizer(self, optimizer_name: str):
        """Get Qiskit optimizer by name."""
        optimizers = {
            "COBYLA": COBYLA(maxiter=self.maxiter),
            "SPSA": SPSA(maxiter=self.maxiter),
            "SLSQP": SLSQP(maxiter=self.maxiter),
        }
        return optimizers.get(optimizer_name, COBYLA(maxiter=self.maxiter))
    
    def solve(
        self,
        qubo_matrix: np.ndarray,
        constant: float = 0.0,
        indexer: Optional[VariableIndexer] = None,
        initial_point: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Solve QUBO using QAOA or fallback to classical methods.
        
        Args:
            qubo_matrix: QUBO Q matrix (n x n)
            constant: Constant offset
            indexer: Variable indexer for decoding
            initial_point: Initial QAOA parameters (gamma, beta) for warm start
            
        Returns:
            Tuple of (optimal bitstring, optimal value)
        """
        n = qubo_matrix.shape[0]
        
        # For problems with > 127 qubits (IBM hardware limit), use classical heuristic
        # Note: Simulator can handle ~20-25 qubits practically, but we allow up to 127 for IBM
        if n > 127:
            print(f"  Problem size ({n} qubits) exceeds IBM hardware limit, using simulated annealing")
            return self.solve_simulated_annealing(qubo_matrix, constant)
        
        # For simulator, practical limit is ~25 qubits due to memory
        if n > 25:
            print(f"  Problem size ({n} qubits) too large for local simulator, using simulated annealing")
            print(f"  (Use --use-ibm-quantum for problems up to 127 qubits)")
            return self.solve_simulated_annealing(qubo_matrix, constant)
        
        # Build QuadraticProgram from QUBO matrix
        qp = self._build_quadratic_program(qubo_matrix, constant, n)
        
        try:
            # Try using QAOA
            from qiskit_aer.primitives import Sampler as AerSampler
            
            sampler = AerSampler(run_options={"shots": self.shots})
            
            qaoa = QAOA(
                sampler=sampler,
                optimizer=self.optimizer,
                reps=self.depth,
                initial_point=initial_point
            )
            
            optimizer = MinimumEigenOptimizer(qaoa)
            self.result = optimizer.solve(qp)
            
            self.optimal_bitstring = np.array([self.result.x[i] for i in range(n)])
            self.optimal_value = self.result.fval
            
            return self.optimal_bitstring, self.optimal_value
            
        except Exception as e:
            print(f"  QAOA failed ({type(e).__name__}), using simulated annealing")
            return self.solve_simulated_annealing(qubo_matrix, constant)
    
    def solve_classical(
        self,
        qubo_matrix: np.ndarray,
        constant: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Solve QUBO using classical simulated annealing.
        
        Args:
            qubo_matrix: QUBO Q matrix
            constant: Constant offset
            
        Returns:
            Tuple of (optimal bitstring, optimal value)
        """
        return self.solve_simulated_annealing(qubo_matrix, constant)
    
    def solve_ibm_quantum(
        self,
        qubo_matrix: np.ndarray,
        constant: float = 0.0,
        backend_name: Optional[str] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Solve QUBO using real IBM Quantum hardware.
        
        Args:
            qubo_matrix: QUBO Q matrix
            constant: Constant offset
            backend_name: IBM Quantum backend (e.g., 'ibm_brisbane', 'ibm_osaka')
                         If None, auto-selects least busy backend
        
        Returns:
            Tuple of (optimal bitstring, optimal value)
        """
        if not IBM_QUANTUM_AVAILABLE:
            print("  IBM Quantum Runtime not available, using simulator")
            return self.solve(qubo_matrix, constant)
        
        # Get IBM Quantum token from environment
        token = os.getenv("IBM_QUANTUM_TOKEN") or os.getenv("IBMQ_TOKEN")
        if not token:
            print("  IBM Quantum token not found in .env, using simulator")
            return self.solve(qubo_matrix, constant)
        
        n = qubo_matrix.shape[0]
        
        # Check problem size (real quantum computers have limited qubits)
        if n > 127:  # Most IBM backends have 127 qubits or less
            print(f"  Problem size ({n} qubits) exceeds IBM hardware limit, using simulator")
            return self.solve(qubo_matrix, constant)
        
        try:
            print(f"  Connecting to IBM Quantum...")
            
            # Initialize IBM Quantum service
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            
            # Select backend
            if backend_name:
                backend = service.backend(backend_name)
            else:
                # Get least busy backend that can handle our problem
                backend = service.least_busy(
                    simulator=False,
                    min_num_qubits=n,
                    operational=True
                )
            
            print(f"  Using IBM Quantum backend: {backend.name}")
            print(f"  Transpiling circuit for hardware...")
            
            # Build QuadraticProgram and convert to Ising Hamiltonian
            qp = self._build_quadratic_program(qubo_matrix, constant, n)
            
            # Convert QUBO to Ising Hamiltonian for QAOA
            from qiskit_optimization.converters import QuadraticProgramToQubo
            from qiskit_optimization.translators import from_docplex_mp
            from qiskit.circuit.library import QAOAAnsatz
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.transpiler import generate_preset_pass_manager
            
            # Get the cost operator (Ising Hamiltonian)
            qubo_converter = QuadraticProgramToQubo()
            qubo = qubo_converter.convert(qp)
            
            # Convert to Ising form 
            linear = {}
            quadratic = {}
            offset = 0.0
            
            # Extract from QUBO objective
            obj = qubo.objective
            offset = obj.constant
            
            for i, coef in obj.linear.to_dict().items():
                linear[i] = coef
            
            for (i, j), coef in obj.quadratic.to_dict().items():
                if i == j:
                    linear[i] = linear.get(i, 0) + coef
                else:
                    quadratic[(i, j)] = coef
            
            # Build cost Hamiltonian as SparsePauliOp
            # H = sum_i h_i Z_i + sum_{ij} J_{ij} Z_i Z_j
            pauli_list = []
            coeff_list = []
            
            for i, h in linear.items():
                if h != 0:
                    pauli_str = ['I'] * n
                    pauli_str[i] = 'Z'
                    pauli_list.append(''.join(pauli_str[::-1]))  # Qiskit uses little-endian
                    coeff_list.append(h / 2)  # QUBO to Ising conversion factor
            
            for (i, j), J in quadratic.items():
                if J != 0:
                    pauli_str = ['I'] * n
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli_list.append(''.join(pauli_str[::-1]))
                    coeff_list.append(J / 4)  # QUBO to Ising conversion factor
            
            if not pauli_list:
                # Trivial Hamiltonian - return zeros
                return np.zeros(n), constant
            
            cost_hamiltonian = SparsePauliOp(pauli_list, coeff_list)
            
            # Create QAOA ansatz
            qaoa_ansatz = QAOAAnsatz(
                cost_operator=cost_hamiltonian,
                reps=self.depth
            )
            
            # Transpile for target backend using generate_preset_pass_manager
            print(f"  Transpiling {n}-qubit QAOA circuit (optimization_level=2)...")
            pass_manager = generate_preset_pass_manager(
                optimization_level=2,
                backend=backend
            )
            transpiled_circuit = pass_manager.run(qaoa_ansatz)
            
            print(f"  Transpiled circuit depth: {transpiled_circuit.depth()}")
            print(f"  Running QAOA optimization on IBM Quantum hardware...")
            print(f"  (This may take 5-30 minutes depending on queue)")
            
            # Create SamplerV2 for IBM execution
            from qiskit_ibm_runtime import SamplerV2
            from qiskit.circuit import ParameterVector
            from scipy.optimize import minimize
            
            sampler = SamplerV2(mode=backend)
            
            # Add measurements to the transpiled circuit
            transpiled_circuit.measure_all()
            
            # Number of parameters: 2 * depth (gamma, beta for each layer)
            num_params = 2 * self.depth
            
            # Cost function for VQE optimization
            iteration_count = [0]
            
            def cost_function(params):
                """Evaluate QAOA cost on IBM hardware."""
                iteration_count[0] += 1
                
                # Bind parameters to circuit
                bound_circuit = transpiled_circuit.assign_parameters(
                    {transpiled_circuit.parameters[i]: params[i] for i in range(len(params))}
                )
                
                # Run on IBM hardware
                job = sampler.run([bound_circuit], shots=self.shots)
                result = job.result()
                
                # Get counts
                pub_result = result[0]
                counts = pub_result.data.meas.get_counts()
                
                # Compute expectation value of cost Hamiltonian
                total_energy = 0.0
                total_shots = sum(counts.values())
                
                for bitstring, count in counts.items():
                    # Convert bitstring to array - take only first n bits (logical qubits)
                    # Transpiled circuit may have more physical qubits
                    full_bits = np.array([int(b) for b in bitstring[::-1]])  # Reverse for little-endian
                    bits = full_bits[:n]  # Take only the logical qubits
                    
                    # Compute QUBO energy: z^T Q z
                    energy = bits @ qubo_matrix @ bits + constant
                    total_energy += energy * count
                
                avg_energy = total_energy / total_shots
                
                if iteration_count[0] % 5 == 0:
                    print(f"    Iteration {iteration_count[0]}: Energy = {avg_energy:.2f}")
                
                return avg_energy
            
            # Initial parameters (random)
            np.random.seed(42)
            initial_params = np.random.uniform(0, 2 * np.pi, num_params)
            
            print(f"  Starting COBYLA optimization ({self.maxiter} max iterations)...")
            
            # Run optimization
            result = minimize(
                cost_function,
                initial_params,
                method='COBYLA',
                options={'maxiter': self.maxiter, 'rhobeg': 0.5}
            )
            
            optimal_params = result.x
            
            # Final measurement with optimal parameters
            print(f"  Running final measurement with optimal parameters...")
            bound_circuit = transpiled_circuit.assign_parameters(
                {transpiled_circuit.parameters[i]: optimal_params[i] for i in range(len(optimal_params))}
            )
            
            job = sampler.run([bound_circuit], shots=self.shots * 2)  # More shots for final
            final_result = job.result()
            counts = final_result[0].data.meas.get_counts()
            
            # Find best bitstring
            best_bitstring = None
            best_energy = float('inf')
            
            for bitstring, count in counts.items():
                bits = np.array([int(b) for b in bitstring[::-1]])[:n]
                energy = bits @ qubo_matrix @ bits + constant
                
                if energy < best_energy:
                    best_energy = energy
                    best_bitstring = bits
            
            self.optimal_bitstring = best_bitstring
            self.optimal_value = best_energy
            
            print(f"  ✅ IBM Quantum execution completed!")
            print(f"  Optimal energy: {best_energy:.2f}")
            print(f"  Total iterations: {iteration_count[0]}")
            
            return self.optimal_bitstring, self.optimal_value
            
        except Exception as e:
            print(f"  IBM Quantum failed ({type(e).__name__}: {e})")
            print("  Falling back to simulator...")
            return self.solve(qubo_matrix, constant)
    
    def solve_simulated_annealing(
        self,
        qubo_matrix: np.ndarray,
        constant: float = 0.0,
        num_reads: int = 100,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95
    ) -> Tuple[np.ndarray, float]:
        """
        Solve QUBO using simulated annealing heuristic.
        
        Args:
            qubo_matrix: QUBO Q matrix
            constant: Constant offset
            num_reads: Number of restarts
            initial_temp: Starting temperature
            cooling_rate: Temperature decay factor
            
        Returns:
            Tuple of (optimal bitstring, optimal value)
        """
        n = qubo_matrix.shape[0]
        
        best_bitstring = None
        best_energy = float('inf')
        
        for _ in range(num_reads):
            # Random initial state
            bitstring = np.random.randint(0, 2, n)
            energy = self._compute_energy(bitstring, qubo_matrix, constant)
            
            temp = initial_temp
            
            # Annealing loop
            for _ in range(n * 50):  # iterations proportional to problem size
                # Flip a random bit
                i = np.random.randint(n)
                new_bitstring = bitstring.copy()
                new_bitstring[i] = 1 - new_bitstring[i]
                
                new_energy = self._compute_energy(new_bitstring, qubo_matrix, constant)
                
                # Accept or reject
                delta = new_energy - energy
                if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                    bitstring = new_bitstring
                    energy = new_energy
                
                # Cool down
                temp *= cooling_rate
            
            # Track best
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring.copy()
        
        self.optimal_bitstring = best_bitstring
        self.optimal_value = best_energy
        
        return best_bitstring, best_energy
    
    def _compute_energy(
        self,
        bitstring: np.ndarray,
        qubo_matrix: np.ndarray,
        constant: float
    ) -> float:
        """Compute QUBO energy E(z) = z^T Q z + const."""
        return float(bitstring @ qubo_matrix @ bitstring + constant)
    
    def _build_quadratic_program(
        self,
        qubo_matrix: np.ndarray,
        constant: float,
        n: int
    ) -> QuadraticProgram:
        """
        Build Qiskit QuadraticProgram from QUBO matrix.
        
        Args:
            qubo_matrix: QUBO Q matrix
            constant: Constant offset
            n: Number of variables
            
        Returns:
            QuadraticProgram object
        """
        qp = QuadraticProgram("supply_chain_stage1")
        
        # Add binary variables
        for i in range(n):
            qp.binary_var(name=f"z_{i}")
        
        # Build objective: minimize z^T Q z + const
        linear = {}
        quadratic = {}
        
        # Diagonal terms (linear in QUBO due to z^2 = z)
        for i in range(n):
            if abs(qubo_matrix[i, i]) > 1e-10:
                linear[f"z_{i}"] = qubo_matrix[i, i]
        
        # Off-diagonal terms (quadratic)
        for i in range(n):
            for j in range(i + 1, n):
                coeff = qubo_matrix[i, j] + qubo_matrix[j, i]
                if abs(coeff) > 1e-10:
                    quadratic[(f"z_{i}", f"z_{j}")] = coeff
        
        qp.minimize(linear=linear, quadratic=quadratic, constant=constant)
        
        return qp
    
    def get_qaoa_parameters(self) -> Optional[np.ndarray]:
        """Get optimized QAOA parameters for warm-start."""
        if self.result is not None and hasattr(self.result, 'min_eigen_solver_result'):
            eigen_result = self.result.min_eigen_solver_result
            if hasattr(eigen_result, 'optimal_point'):
                return eigen_result.optimal_point
        return None
    
    def get_solution_samples(self, top_k: int = 10) -> List[Tuple[np.ndarray, float, float]]:
        """Get top-k solution samples."""
        if self.result is None:
            return []
        
        samples = []
        if hasattr(self.result, 'samples'):
            for sample in self.result.samples[:top_k]:
                bitstring = np.array(list(sample.x.values()))
                samples.append((bitstring, sample.fval, sample.probability))
        
        return samples


def run_qaoa_stage1(
    qubo_builder: QUBOBuilder,
    config: QAOAConfig,
    use_classical: bool = False
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], float]:
    """
    Convenience function to run Stage 1 QAOA.
    
    Args:
        qubo_builder: Built QUBO builder
        config: QAOA configuration
        use_classical: If True, use classical solver instead of QAOA
        
    Returns:
        Tuple of (x_assignments, y_assignments, optimal_value)
    """
    qubo_matrix, constant, indexer = qubo_builder.build()
    
    solver = QAOASolver(config)
    
    if use_classical:
        bitstring, value = solver.solve_classical(qubo_matrix, constant)
    else:
        bitstring, value = solver.solve(qubo_matrix, constant, indexer)
    
    x_assign, y_assign = qubo_builder.decode_solution(bitstring)
    
    return x_assign, y_assign, value
