"""
AI-Powered Quantum Circuit Optimizer
====================================

Machine learning-based optimization for quantum circuits,
including noise-aware compilation and parameter optimization.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from circuit optimization."""
    original_circuit: Any
    optimized_circuit: Any
    original_depth: int
    optimized_depth: int
    original_gate_count: int
    optimized_gate_count: int
    improvement_percent: float
    optimizations_applied: List[str] = field(default_factory=list)
    fidelity_estimate: float = 1.0


class AIOptimizer:
    """
    AI-powered quantum circuit optimizer.
    
    Uses machine learning techniques and heuristics to:
    - Reduce circuit depth
    - Minimize gate count
    - Optimize for specific hardware constraints
    - Find optimal variational parameters
    
    Example:
        >>> optimizer = AIOptimizer()
        >>> result = optimizer.optimize(circuit)
        >>> print(f"Improved by {result.improvement_percent:.1f}%")
    """
    
    # Gate cancellation rules
    CANCELLATION_RULES = {
        ("h", "h"): [],  # H·H = I
        ("x", "x"): [],  # X·X = I
        ("y", "y"): [],  # Y·Y = I
        ("z", "z"): [],  # Z·Z = I
        ("s", "s"): [("z",)],  # S·S = Z
        ("t", "t"): [("s",)],  # T·T = S
        ("s", "sdg"): [],  # S·S† = I
        ("t", "tdg"): [],  # T·T† = I
        ("cnot", "cnot"): [],  # CNOT·CNOT = I
        ("cz", "cz"): [],  # CZ·CZ = I
        ("swap", "swap"): [],  # SWAP·SWAP = I
    }
    
    # Gate commutation rules (gates that can be swapped)
    COMMUTATION_RULES = {
        "z": {"z", "s", "t", "rz", "cz"},  # Z commutes with phase gates
        "x": {"x", "rx", "cnot"},  # X-type gates
        "h": set(),  # H doesn't commute with much
    }
    
    def __init__(
        self,
        use_ml: bool = True,
        target_hardware: Optional[str] = None,
    ):
        """
        Initialize the optimizer.
        
        Args:
            use_ml: Whether to use ML-based optimization
            target_hardware: Target hardware topology (e.g., "linear", "grid", "all-to-all")
        """
        self.use_ml = use_ml
        self.target_hardware = target_hardware
        
        # Try to import torch for ML optimization
        self._torch_available = False
        if use_ml:
            try:
                import torch
                self._torch = torch
                self._torch_available = True
            except ImportError:
                logger.warning("PyTorch not available. ML optimization disabled.")
    
    def optimize(
        self,
        circuit: Any,
        max_iterations: int = 100,
        target_depth: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Optimize a quantum circuit.
        
        Args:
            circuit: Circuit to optimize
            max_iterations: Maximum optimization iterations
            target_depth: Target circuit depth (optional)
            
        Returns:
            OptimizationResult with original and optimized circuits
        """
        from quantumviz.core.simulator import QuantumCircuit
        
        # Create a copy of the circuit
        optimized = QuantumCircuit(circuit.num_qubits, name=f"{circuit.name}_optimized")
        optimized.operations = circuit.operations.copy()
        
        original_depth = circuit.depth
        original_gate_count = circuit.gate_count
        
        optimizations_applied = []
        
        # Apply optimization passes
        for iteration in range(max_iterations):
            changed = False
            
            # Pass 1: Gate cancellation
            new_ops, cancelled = self._cancel_gates(optimized.operations)
            if cancelled:
                optimized.operations = new_ops
                optimizations_applied.append("gate_cancellation")
                changed = True
            
            # Pass 2: Gate merging
            new_ops, merged = self._merge_rotations(optimized.operations)
            if merged:
                optimized.operations = new_ops
                optimizations_applied.append("rotation_merging")
                changed = True
            
            # Pass 3: Commutation for depth reduction
            new_ops, commuted = self._commute_gates(optimized.operations)
            if commuted:
                optimized.operations = new_ops
                optimizations_applied.append("gate_commutation")
                changed = True
            
            if not changed:
                break
            
            # Check if we reached target depth
            if target_depth and self._calculate_depth(optimized.operations, circuit.num_qubits) <= target_depth:
                break
        
        # Pass 4: Peephole optimization
        new_ops = self._peephole_optimize(optimized.operations)
        if len(new_ops) < len(optimized.operations):
            optimized.operations = new_ops
            optimizations_applied.append("peephole_optimization")
        
        optimized_depth = optimized.depth
        optimized_gate_count = optimized.gate_count
        
        improvement = 0.0
        if original_gate_count > 0:
            improvement = (original_gate_count - optimized_gate_count) / original_gate_count * 100
        
        return OptimizationResult(
            original_circuit=circuit,
            optimized_circuit=optimized,
            original_depth=original_depth,
            optimized_depth=optimized_depth,
            original_gate_count=original_gate_count,
            optimized_gate_count=optimized_gate_count,
            improvement_percent=improvement,
            optimizations_applied=list(set(optimizations_applied)),
        )
    
    def _cancel_gates(self, operations: List[Tuple]) -> Tuple[List[Tuple], bool]:
        """Cancel adjacent inverse gates."""
        if len(operations) < 2:
            return operations, False
        
        new_ops = []
        i = 0
        changed = False
        
        while i < len(operations):
            if i < len(operations) - 1:
                op1 = operations[i]
                op2 = operations[i + 1]
                
                # Check if same gate on same qubits
                if (op1[0], op1[1:]) == (op2[0], op2[1:]):
                    # Check cancellation rules
                    key = (op1[0], op2[0])
                    if key in self.CANCELLATION_RULES:
                        replacement = self.CANCELLATION_RULES[key]
                        for rep in replacement:
                            new_ops.append((rep[0],) + op1[1:])
                        i += 2
                        changed = True
                        continue
                    
                    # Self-inverse gates
                    if op1[0] in ["h", "x", "y", "z", "cnot", "cz", "swap"]:
                        i += 2
                        changed = True
                        continue
            
            new_ops.append(operations[i])
            i += 1
        
        return new_ops, changed
    
    def _merge_rotations(self, operations: List[Tuple]) -> Tuple[List[Tuple], bool]:
        """Merge adjacent rotation gates."""
        if len(operations) < 2:
            return operations, False
        
        new_ops = []
        i = 0
        changed = False
        
        while i < len(operations):
            if i < len(operations) - 1:
                op1 = operations[i]
                op2 = operations[i + 1]
                
                # Check if both are rotations on same qubit and axis
                if op1[0] in ["rx", "ry", "rz"] and op1[0] == op2[0]:
                    if len(op1) >= 3 and len(op2) >= 3 and op1[1] == op2[1]:
                        # Merge angles
                        merged_angle = op1[2] + op2[2]
                        # Reduce angle to [-2π, 2π]
                        while merged_angle > 2 * np.pi:
                            merged_angle -= 2 * np.pi
                        while merged_angle < -2 * np.pi:
                            merged_angle += 2 * np.pi
                        
                        if abs(merged_angle) > 1e-10:
                            new_ops.append((op1[0], op1[1], merged_angle))
                        # If angle is ~0, skip the gate entirely
                        
                        i += 2
                        changed = True
                        continue
            
            new_ops.append(operations[i])
            i += 1
        
        return new_ops, changed
    
    def _commute_gates(self, operations: List[Tuple]) -> Tuple[List[Tuple], bool]:
        """Commute gates to reduce depth."""
        if len(operations) < 2:
            return operations, False
        
        new_ops = operations.copy()
        changed = False
        
        # Simple bubble sort style commutation
        for _ in range(len(operations)):
            swapped = False
            for i in range(len(new_ops) - 1):
                op1 = new_ops[i]
                op2 = new_ops[i + 1]
                
                # Check if gates are on different qubits (can definitely commute)
                qubits1 = set(op1[1:] if len(op1) > 1 else [])
                qubits2 = set(op2[1:] if len(op2) > 1 else [])
                
                if not isinstance(list(qubits1)[0] if qubits1 else 0, int):
                    qubits1 = {op1[1]} if len(op1) > 1 else set()
                if not isinstance(list(qubits2)[0] if qubits2 else 0, int):
                    qubits2 = {op2[1]} if len(op2) > 1 else set()
                
                if qubits1.isdisjoint(qubits2):
                    # Can swap, but only if it helps (smaller qubit indices first)
                    min1 = min(qubits1) if qubits1 else 0
                    min2 = min(qubits2) if qubits2 else 0
                    
                    if min2 < min1:
                        new_ops[i], new_ops[i + 1] = new_ops[i + 1], new_ops[i]
                        swapped = True
                        changed = True
            
            if not swapped:
                break
        
        return new_ops, changed
    
    def _peephole_optimize(self, operations: List[Tuple]) -> List[Tuple]:
        """Apply peephole optimizations (local pattern matching)."""
        patterns = [
            # H-Z-H = X
            (["h", "z", "h"], ["x"]),
            # H-X-H = Z
            (["h", "x", "h"], ["z"]),
            # S-S = Z
            (["s", "s"], ["z"]),
            # T-T-T-T = Z
            (["t", "t", "t", "t"], ["z"]),
            # X-X = I
            (["x", "x"], []),
            # Y-Y = I
            (["y", "y"], []),
            # Z-Z = I
            (["z", "z"], []),
            # H-H = I
            (["h", "h"], []),
        ]
        
        new_ops = operations.copy()
        
        for pattern, replacement in patterns:
            i = 0
            while i <= len(new_ops) - len(pattern):
                # Check if pattern matches
                match = True
                qubit = None
                
                for j, gate in enumerate(pattern):
                    if new_ops[i + j][0] != gate:
                        match = False
                        break
                    
                    op_qubit = new_ops[i + j][1] if len(new_ops[i + j]) > 1 else None
                    if qubit is None:
                        qubit = op_qubit
                    elif op_qubit != qubit:
                        match = False
                        break
                
                if match and qubit is not None:
                    # Replace pattern
                    new_ops = (
                        new_ops[:i] +
                        [(gate, qubit) for gate in replacement] +
                        new_ops[i + len(pattern):]
                    )
                    # Don't increment i to check for overlapping patterns
                else:
                    i += 1
        
        return new_ops
    
    def _calculate_depth(self, operations: List[Tuple], num_qubits: int) -> int:
        """Calculate circuit depth."""
        if not operations:
            return 0
        
        qubit_depths = [0] * num_qubits
        
        for op in operations:
            if len(op) < 2:
                continue
            
            # Get qubits involved
            qubits = []
            for item in op[1:]:
                if isinstance(item, int):
                    qubits.append(item)
            
            if not qubits:
                continue
            
            # Find max depth among involved qubits
            max_depth = max(qubit_depths[q] for q in qubits if q < num_qubits)
            new_depth = max_depth + 1
            
            # Update depths
            for q in qubits:
                if q < num_qubits:
                    qubit_depths[q] = new_depth
        
        return max(qubit_depths) if qubit_depths else 0
    
    def optimize_parameters(
        self,
        circuit_builder: Callable[[np.ndarray], Any],
        cost_function: Callable[[Any], float],
        initial_params: np.ndarray,
        method: str = "adam",
        max_iterations: int = 100,
        learning_rate: float = 0.1,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize variational circuit parameters.
        
        Args:
            circuit_builder: Function that builds circuit from parameters
            cost_function: Function that evaluates circuit cost
            initial_params: Initial parameter values
            method: Optimization method ("adam", "sgd", "spsa")
            max_iterations: Maximum iterations
            learning_rate: Learning rate
            
        Returns:
            Tuple of (optimized_params, cost_history)
        """
        params = initial_params.copy()
        cost_history = []
        
        if method == "adam" and self._torch_available:
            return self._optimize_adam(
                circuit_builder, cost_function, params,
                max_iterations, learning_rate
            )
        elif method == "spsa":
            return self._optimize_spsa(
                circuit_builder, cost_function, params,
                max_iterations, learning_rate
            )
        else:
            return self._optimize_gradient_descent(
                circuit_builder, cost_function, params,
                max_iterations, learning_rate
            )
    
    def _optimize_adam(
        self,
        circuit_builder: Callable,
        cost_function: Callable,
        params: np.ndarray,
        max_iterations: int,
        learning_rate: float,
    ) -> Tuple[np.ndarray, List[float]]:
        """Optimize using Adam optimizer with PyTorch."""
        torch = self._torch
        
        params_tensor = torch.tensor(params, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([params_tensor], lr=learning_rate)
        
        cost_history = []
        
        for _ in range(max_iterations):
            optimizer.zero_grad()
            
            # Compute cost with gradient
            circuit = circuit_builder(params_tensor.detach().numpy())
            cost = cost_function(circuit)
            
            cost_history.append(float(cost))
            
            # Numerical gradient
            eps = 1e-5
            grads = np.zeros_like(params)
            for i in range(len(params)):
                params_plus = params_tensor.detach().numpy().copy()
                params_plus[i] += eps
                params_minus = params_tensor.detach().numpy().copy()
                params_minus[i] -= eps
                
                cost_plus = cost_function(circuit_builder(params_plus))
                cost_minus = cost_function(circuit_builder(params_minus))
                
                grads[i] = (cost_plus - cost_minus) / (2 * eps)
            
            params_tensor.grad = torch.tensor(grads, dtype=torch.float32)
            optimizer.step()
            
            # Convergence check
            if len(cost_history) > 10:
                if abs(cost_history[-1] - cost_history[-10]) < 1e-6:
                    break
        
        return params_tensor.detach().numpy(), cost_history
    
    def _optimize_spsa(
        self,
        circuit_builder: Callable,
        cost_function: Callable,
        params: np.ndarray,
        max_iterations: int,
        learning_rate: float,
    ) -> Tuple[np.ndarray, List[float]]:
        """Simultaneous Perturbation Stochastic Approximation."""
        cost_history = []
        
        a = learning_rate
        c = 0.1
        A = max_iterations // 10
        alpha = 0.602
        gamma = 0.101
        
        for k in range(max_iterations):
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma
            
            # Random direction
            delta = 2 * np.random.randint(0, 2, size=params.shape) - 1
            
            # Evaluate at perturbed points
            params_plus = params + ck * delta
            params_minus = params - ck * delta
            
            cost_plus = cost_function(circuit_builder(params_plus))
            cost_minus = cost_function(circuit_builder(params_minus))
            
            # Gradient estimate
            grad = (cost_plus - cost_minus) / (2 * ck * delta)
            
            # Update
            params = params - ak * grad
            
            # Record cost
            cost = cost_function(circuit_builder(params))
            cost_history.append(cost)
            
            if len(cost_history) > 10:
                if abs(cost_history[-1] - cost_history[-10]) < 1e-6:
                    break
        
        return params, cost_history
    
    def _optimize_gradient_descent(
        self,
        circuit_builder: Callable,
        cost_function: Callable,
        params: np.ndarray,
        max_iterations: int,
        learning_rate: float,
    ) -> Tuple[np.ndarray, List[float]]:
        """Simple gradient descent with numerical gradients."""
        cost_history = []
        eps = 1e-5
        
        for _ in range(max_iterations):
            # Compute gradient
            grads = np.zeros_like(params)
            
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                params_minus = params.copy()
                params_minus[i] -= eps
                
                cost_plus = cost_function(circuit_builder(params_plus))
                cost_minus = cost_function(circuit_builder(params_minus))
                
                grads[i] = (cost_plus - cost_minus) / (2 * eps)
            
            # Update
            params = params - learning_rate * grads
            
            # Record cost
            cost = cost_function(circuit_builder(params))
            cost_history.append(cost)
            
            if len(cost_history) > 10:
                if abs(cost_history[-1] - cost_history[-10]) < 1e-6:
                    break
        
        return params, cost_history
    
    def suggest_hardware_mapping(
        self,
        circuit: Any,
        topology: str = "linear",
    ) -> Dict[str, Any]:
        """
        Suggest qubit mapping for target hardware topology.
        
        Args:
            circuit: Circuit to map
            topology: Target topology ("linear", "grid", "all-to-all")
            
        Returns:
            Dictionary with mapping suggestions
        """
        num_qubits = circuit.num_qubits
        
        # Analyze two-qubit gate connectivity
        connectivity = {}
        for op in circuit.operations:
            if op[0] in ["cnot", "cz", "swap", "cx"]:
                q1, q2 = op[1], op[2]
                key = tuple(sorted([q1, q2]))
                connectivity[key] = connectivity.get(key, 0) + 1
        
        # Sort by frequency
        sorted_connections = sorted(
            connectivity.items(),
            key=lambda x: -x[1]
        )
        
        suggestions = {
            "topology": topology,
            "num_qubits": num_qubits,
            "two_qubit_gates": sum(connectivity.values()),
            "unique_connections": len(connectivity),
            "most_connected_pairs": sorted_connections[:5],
        }
        
        if topology == "linear":
            # Linear topology: qubits i and i+1 are connected
            # Suggest placing most-connected pairs adjacent
            mapping = list(range(num_qubits))
            
            if sorted_connections:
                # Place most connected pair at center
                q1, q2 = sorted_connections[0][0]
                center = num_qubits // 2
                mapping[q1], mapping[center] = mapping[center], mapping[q1]
                mapping[q2], mapping[center + 1] = mapping[center + 1], mapping[q2]
            
            suggestions["suggested_mapping"] = mapping
            
            # Calculate SWAP count needed
            swap_count = 0
            for (q1, q2), count in sorted_connections:
                if abs(mapping.index(q1) - mapping.index(q2)) > 1:
                    swap_count += count
            
            suggestions["estimated_swap_overhead"] = swap_count
        
        elif topology == "grid":
            # 2D grid topology
            import math
            side = math.ceil(math.sqrt(num_qubits))
            suggestions["grid_size"] = (side, side)
            suggestions["suggested_mapping"] = list(range(num_qubits))
        
        else:  # all-to-all
            suggestions["suggested_mapping"] = list(range(num_qubits))
            suggestions["estimated_swap_overhead"] = 0
        
        return suggestions
