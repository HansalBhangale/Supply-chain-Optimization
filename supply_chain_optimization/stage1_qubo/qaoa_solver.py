"""
QAOA solver integration with IBM Qiskit.

Solves Stage 1 QUBO using Quantum Approximate Optimization Algorithm.
"""

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
        
        # For problems with > 20 qubits, use classical heuristic
        if n > 20:
            print(f"  Problem size ({n} qubits) too large for QAOA, using simulated annealing")
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
