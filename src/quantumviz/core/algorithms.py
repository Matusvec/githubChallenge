"""
Quantum Algorithms Library
==========================

Implementation of famous quantum algorithms for visualization:
- Grover's Search Algorithm
- Quantum Fourier Transform (QFT)
- Quantum Phase Estimation (QPE)
- Variational Quantum Eigensolver (VQE)
- QAOA (Quantum Approximate Optimization)
- Quantum Walk
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass

from quantumviz.core.simulator import QuantumCircuit, QuantumSimulator, SimulationResult
from quantumviz.core.quantum_state import QuantumState


@dataclass
class AlgorithmResult:
    """Result from running a quantum algorithm."""
    name: str
    circuit: QuantumCircuit
    simulation_result: SimulationResult
    iterations: int = 1
    parameters: Dict[str, Any] = None
    success_probability: float = 0.0
    classical_result: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.name,
            "iterations": self.iterations,
            "parameters": self.parameters,
            "success_probability": self.success_probability,
            "classical_result": self.classical_result,
            "simulation": self.simulation_result.to_dict() if self.simulation_result else None,
        }


class QuantumAlgorithms:
    """
    Library of quantum algorithm implementations.
    
    Each algorithm returns both the circuit and execution results,
    optimized for visualization purposes.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GROVER'S SEARCH ALGORITHM
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def grover_search(
        num_qubits: int,
        target_states: List[int],
        num_iterations: Optional[int] = None,
        oracle_type: str = "phase"
    ) -> Tuple[QuantumCircuit, int]:
        """
        Build Grover's search algorithm circuit.
        
        Searches for target states in an unsorted database of N = 2^n items.
        Provides quadratic speedup: O(√N) vs O(N) classical.
        
        Args:
            num_qubits: Number of qubits (database size = 2^n)
            target_states: List of target state indices to find
            num_iterations: Number of Grover iterations (auto-calculated if None)
            oracle_type: "phase" or "boolean"
            
        Returns:
            Tuple of (circuit, optimal_iterations)
        """
        n = num_qubits
        N = 2 ** n
        M = len(target_states)
        
        # Optimal number of iterations
        if num_iterations is None:
            num_iterations = int(np.round(np.pi / 4 * np.sqrt(N / M)))
            num_iterations = max(1, num_iterations)
        
        circuit = QuantumCircuit(n, n, name=f"Grover_{n}qubits")
        
        # Initialize superposition
        for i in range(n):
            circuit.h(i)
        
        # Apply Grover iterations
        for iteration in range(num_iterations):
            # Oracle: Mark target states with phase flip
            circuit.barrier()
            QuantumAlgorithms._add_grover_oracle(circuit, n, target_states)
            
            # Diffusion operator: Inversion about average
            circuit.barrier()
            QuantumAlgorithms._add_diffusion_operator(circuit, n)
        
        # Measurement
        circuit.barrier()
        circuit.measure_all()
        
        return circuit, num_iterations
    
    @staticmethod
    def _add_grover_oracle(circuit: QuantumCircuit, n: int, targets: List[int]):
        """Add oracle that marks target states with phase flip."""
        for target in targets:
            # Convert target to binary and apply X gates for 0s
            target_bits = format(target, f'0{n}b')
            
            # Apply X gates to flip 0s to 1s
            for i, bit in enumerate(target_bits):
                if bit == '0':
                    circuit.x(i)
            
            # Multi-controlled Z gate (implemented via H-MCX-H on last qubit)
            if n == 1:
                circuit.z(0)
            elif n == 2:
                circuit.cz(0, 1)
            elif n == 3:
                circuit.h(n - 1)
                circuit.ccx(0, 1, 2)
                circuit.h(n - 1)
            else:
                # Decompose into smaller controlled gates for larger circuits
                circuit.h(n - 1)
                # Simplified Toffoli cascade
                for i in range(n - 2):
                    circuit.ccx(i, i + 1, n - 1)
                circuit.h(n - 1)
            
            # Undo X gates
            for i, bit in enumerate(target_bits):
                if bit == '0':
                    circuit.x(i)
    
    @staticmethod
    def _add_diffusion_operator(circuit: QuantumCircuit, n: int):
        """Add diffusion operator (inversion about average)."""
        # H gates
        for i in range(n):
            circuit.h(i)
        
        # X gates
        for i in range(n):
            circuit.x(i)
        
        # Multi-controlled Z
        if n == 1:
            circuit.z(0)
        elif n == 2:
            circuit.cz(0, 1)
        elif n == 3:
            circuit.h(n - 1)
            circuit.ccx(0, 1, 2)
            circuit.h(n - 1)
        else:
            circuit.h(n - 1)
            for i in range(n - 2):
                circuit.ccx(i, i + 1, n - 1)
            circuit.h(n - 1)
        
        # X gates
        for i in range(n):
            circuit.x(i)
        
        # H gates
        for i in range(n):
            circuit.h(i)
    
    @staticmethod
    def run_grover(
        num_qubits: int,
        target_states: List[int],
        shots: int = 1000,
        noise_model=None
    ) -> AlgorithmResult:
        """
        Execute Grover's algorithm and return results.
        
        Args:
            num_qubits: Search space size (2^n items)
            target_states: Target states to find
            shots: Number of measurement shots
            noise_model: Optional noise model
            
        Returns:
            AlgorithmResult with circuit and results
        """
        circuit, num_iterations = QuantumAlgorithms.grover_search(num_qubits, target_states)
        
        simulator = QuantumSimulator()
        if noise_model:
            simulator.set_noise_model(noise_model)
        simulator.enable_history_capture(True)
        
        result = simulator.run(circuit, shots=shots)
        
        # Calculate success probability
        total_target_counts = 0
        for target in target_states:
            target_str = format(target, f'0{num_qubits}b')
            total_target_counts += result.counts.get(target_str, 0) if result.counts else 0
        
        success_prob = total_target_counts / shots if shots > 0 else 0.0
        
        # Find most likely result
        if result.counts:
            best_result = max(result.counts.items(), key=lambda x: x[1])
            classical_result = int(best_result[0], 2)
        else:
            classical_result = None
        
        return AlgorithmResult(
            name="Grover's Search",
            circuit=circuit,
            simulation_result=result,
            iterations=num_iterations,
            parameters={
                "num_qubits": num_qubits,
                "target_states": target_states,
                "search_space_size": 2 ** num_qubits,
            },
            success_probability=success_prob,
            classical_result=classical_result
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUANTUM FOURIER TRANSFORM
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def qft(num_qubits: int, inverse: bool = False, swap: bool = True) -> QuantumCircuit:
        """
        Build Quantum Fourier Transform circuit.
        
        QFT is the quantum analog of discrete Fourier transform, essential
        for algorithms like Shor's and phase estimation.
        
        Args:
            num_qubits: Number of qubits
            inverse: If True, build inverse QFT
            swap: If True, include final SWAP gates for correct ordering
            
        Returns:
            QFT circuit
        """
        n = num_qubits
        name = f"{'Inverse_' if inverse else ''}QFT_{n}qubits"
        circuit = QuantumCircuit(n, n, name=name)
        
        if inverse:
            # Inverse QFT: Reverse order of operations
            if swap:
                for i in range(n // 2):
                    circuit.swap(i, n - 1 - i)
            
            for i in range(n - 1, -1, -1):
                for j in range(n - 1, i, -1):
                    angle = -np.pi / (2 ** (j - i))
                    circuit.cp(angle, j, i)
                circuit.h(i)
        else:
            # Forward QFT
            for i in range(n):
                circuit.h(i)
                for j in range(i + 1, n):
                    angle = np.pi / (2 ** (j - i))
                    circuit.cp(angle, j, i)
            
            if swap:
                for i in range(n // 2):
                    circuit.swap(i, n - 1 - i)
        
        return circuit
    
    @staticmethod
    def run_qft(num_qubits: int, input_state: Optional[str] = None) -> AlgorithmResult:
        """
        Execute QFT on an input state.
        
        Args:
            num_qubits: Number of qubits
            input_state: Optional binary string for initial state (default: |0...0⟩)
        """
        circuit = QuantumAlgorithms.qft(num_qubits)
        
        # Create initial state
        if input_state:
            # Prepare initial state
            init_circuit = QuantumCircuit(num_qubits, name="init")
            for i, bit in enumerate(input_state):
                if bit == '1':
                    init_circuit.x(i)
            
            # Combine with QFT
            combined = QuantumCircuit(num_qubits, name=f"QFT_on_{input_state}")
            for instr in init_circuit.instructions:
                combined._add_instruction(instr.gate_name, instr.target_qubits, instr.params)
            for instr in circuit.instructions:
                combined._add_instruction(instr.gate_name, instr.target_qubits, instr.params)
            circuit = combined
        
        circuit.measure_all()
        
        simulator = QuantumSimulator()
        simulator.enable_history_capture(True)
        result = simulator.run(circuit, shots=1000)
        
        return AlgorithmResult(
            name="Quantum Fourier Transform",
            circuit=circuit,
            simulation_result=result,
            parameters={"num_qubits": num_qubits, "input_state": input_state}
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUANTUM PHASE ESTIMATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def phase_estimation(
        num_counting_qubits: int,
        unitary: np.ndarray,
        eigenstate_prep: Optional[Callable[[QuantumCircuit], None]] = None
    ) -> QuantumCircuit:
        """
        Build Quantum Phase Estimation circuit.
        
        Estimates eigenvalue e^{2πiθ} of a unitary operator U.
        Fundamental subroutine in many quantum algorithms.
        
        Args:
            num_counting_qubits: Precision of phase estimation
            unitary: 2x2 unitary matrix (single-qubit gate)
            eigenstate_prep: Optional function to prepare eigenstate
            
        Returns:
            QPE circuit
        """
        n_count = num_counting_qubits
        n_target = 1  # Single qubit target for simplicity
        n_total = n_count + n_target
        
        circuit = QuantumCircuit(n_total, n_count, name=f"QPE_{n_count}bits")
        target = n_count  # Target qubit index
        
        # Prepare counting qubits in superposition
        for i in range(n_count):
            circuit.h(i)
        
        # Prepare eigenstate (default: |1⟩ which is eigenstate of many gates)
        if eigenstate_prep:
            eigenstate_prep(circuit)
        else:
            circuit.x(target)
        
        # Controlled-U operations
        for i in range(n_count):
            power = 2 ** (n_count - 1 - i)
            # Apply U^(2^i) controlled by qubit i
            # For a phase gate P(θ), this becomes P(θ * 2^i)
            
            # Extract phase from unitary (assuming diagonal or rotation)
            phase = np.angle(unitary[1, 1])
            circuit.cp(phase * power, i, target)
        
        # Inverse QFT on counting qubits
        for i in range(n_count - 1, -1, -1):
            for j in range(n_count - 1, i, -1):
                angle = -np.pi / (2 ** (j - i))
                circuit.cp(angle, j, i)
            circuit.h(i)
        
        # Swap to correct ordering
        for i in range(n_count // 2):
            circuit.swap(i, n_count - 1 - i)
        
        # Measure counting qubits
        for i in range(n_count):
            circuit.measure(i, i)
        
        return circuit
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VARIATIONAL QUANTUM EIGENSOLVER (VQE)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def vqe_ansatz(
        num_qubits: int,
        params: List[float],
        layers: int = 1,
        entanglement: str = "linear"
    ) -> QuantumCircuit:
        """
        Build VQE variational ansatz circuit.
        
        Hardware-efficient ansatz with rotation and entanglement layers.
        
        Args:
            num_qubits: Number of qubits
            params: Variational parameters [θ₁, θ₂, ...]
            layers: Number of variational layers
            entanglement: "linear", "circular", or "full"
            
        Returns:
            Parameterized ansatz circuit
        """
        n = num_qubits
        circuit = QuantumCircuit(n, n, name=f"VQE_ansatz_L{layers}")
        
        param_idx = 0
        
        for layer in range(layers):
            # Rotation layer: RY and RZ on each qubit
            for i in range(n):
                if param_idx < len(params):
                    circuit.ry(params[param_idx], i)
                    param_idx += 1
                if param_idx < len(params):
                    circuit.rz(params[param_idx], i)
                    param_idx += 1
            
            # Entanglement layer
            if entanglement == "linear":
                for i in range(n - 1):
                    circuit.cx(i, i + 1)
            elif entanglement == "circular":
                for i in range(n - 1):
                    circuit.cx(i, i + 1)
                if n > 2:
                    circuit.cx(n - 1, 0)
            elif entanglement == "full":
                for i in range(n):
                    for j in range(i + 1, n):
                        circuit.cx(i, j)
        
        # Final rotation layer
        for i in range(n):
            if param_idx < len(params):
                circuit.ry(params[param_idx], i)
                param_idx += 1
        
        return circuit
    
    @staticmethod
    def run_vqe(
        hamiltonian: np.ndarray,
        num_qubits: int,
        layers: int = 2,
        max_iterations: int = 100,
        learning_rate: float = 0.1
    ) -> AlgorithmResult:
        """
        Run VQE to find ground state energy.
        
        Uses gradient descent to minimize ⟨ψ(θ)|H|ψ(θ)⟩.
        
        Args:
            hamiltonian: Hamiltonian matrix
            num_qubits: Number of qubits
            layers: Ansatz layers
            max_iterations: Maximum optimization iterations
            learning_rate: Gradient descent step size
            
        Returns:
            AlgorithmResult with optimized parameters and energy
        """
        # Number of parameters: 3 * n * layers (RY, RZ per qubit per layer + final RY)
        num_params = 2 * num_qubits * layers + num_qubits
        params = np.random.uniform(-np.pi, np.pi, num_params)
        
        simulator = QuantumSimulator()
        
        def energy(p):
            """Compute expectation value of Hamiltonian."""
            circuit = QuantumAlgorithms.vqe_ansatz(num_qubits, p.tolist(), layers)
            result = simulator.run(circuit)
            return np.real(simulator.get_expectation_value(result.final_state, hamiltonian))
        
        # Simple gradient descent
        energies = []
        for i in range(max_iterations):
            current_energy = energy(params)
            energies.append(current_energy)
            
            # Compute gradient via parameter shift rule
            gradient = np.zeros_like(params)
            shift = np.pi / 2
            
            for j in range(len(params)):
                params_plus = params.copy()
                params_plus[j] += shift
                params_minus = params.copy()
                params_minus[j] -= shift
                
                gradient[j] = (energy(params_plus) - energy(params_minus)) / 2
            
            params -= learning_rate * gradient
            
            # Check convergence
            if i > 0 and abs(energies[-1] - energies[-2]) < 1e-6:
                break
        
        # Final circuit and result
        final_circuit = QuantumAlgorithms.vqe_ansatz(num_qubits, params.tolist(), layers)
        final_circuit.measure_all()
        
        simulator.enable_history_capture(True)
        final_result = simulator.run(final_circuit, shots=1000)
        
        return AlgorithmResult(
            name="Variational Quantum Eigensolver",
            circuit=final_circuit,
            simulation_result=final_result,
            iterations=len(energies),
            parameters={
                "num_qubits": num_qubits,
                "layers": layers,
                "final_params": params.tolist(),
                "energy_history": energies,
            },
            classical_result=energies[-1]  # Ground state energy estimate
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUANTUM WALK
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def quantum_walk_1d(num_positions: int, num_steps: int) -> QuantumCircuit:
        """
        Build 1D discrete quantum walk circuit.
        
        Quantum analog of classical random walk with interference effects.
        
        Args:
            num_positions: Number of position states (power of 2)
            num_steps: Number of walk steps
            
        Returns:
            Quantum walk circuit
        """
        n_pos = int(np.ceil(np.log2(num_positions)))  # Position qubits
        n_coin = 1  # Coin qubit
        n_total = n_pos + n_coin
        
        circuit = QuantumCircuit(n_total, n_total, name=f"QWalk_1D_{num_steps}steps")
        
        coin = 0  # Coin qubit index
        pos_qubits = list(range(1, n_total))  # Position qubit indices
        
        # Initialize: Coin in superposition, position at center
        circuit.h(coin)
        
        # Initialize position to center
        center = num_positions // 2
        center_bits = format(center, f'0{n_pos}b')
        for i, bit in enumerate(center_bits):
            if bit == '1':
                circuit.x(pos_qubits[i])
        
        # Quantum walk steps
        for step in range(num_steps):
            # Coin flip (Hadamard)
            circuit.h(coin)
            
            # Conditional shift
            # If coin = |0⟩: shift left (decrement position)
            # If coin = |1⟩: shift right (increment position)
            
            # Simplified shift using controlled operations
            for i in range(n_pos):
                # Controlled increment/decrement
                circuit.cx(coin, pos_qubits[i])
        
        circuit.measure_all()
        return circuit
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BELL STATE PREPARATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def bell_state_circuit(state_type: str = "phi+") -> QuantumCircuit:
        """
        Create circuit for Bell state preparation.
        
        Args:
            state_type: "phi+", "phi-", "psi+", or "psi-"
            
        Returns:
            Bell state circuit
        """
        circuit = QuantumCircuit(2, 2, name=f"Bell_{state_type}")
        
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        circuit.h(0)
        circuit.cx(0, 1)
        
        if state_type == "phi-":  # (|00⟩ - |11⟩)/√2
            circuit.z(0)
        elif state_type == "psi+":  # (|01⟩ + |10⟩)/√2
            circuit.x(1)
        elif state_type == "psi-":  # (|01⟩ - |10⟩)/√2
            circuit.x(1)
            circuit.z(0)
        
        return circuit
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GHZ STATE PREPARATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def ghz_state_circuit(num_qubits: int) -> QuantumCircuit:
        """
        Create circuit for GHZ state: (|00...0⟩ + |11...1⟩)/√2
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            GHZ state circuit
        """
        circuit = QuantumCircuit(num_qubits, num_qubits, name=f"GHZ_{num_qubits}")
        
        circuit.h(0)
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
        
        return circuit
    
    @staticmethod
    def w_state_circuit(num_qubits: int) -> QuantumCircuit:
        """
        Create circuit for W state: (|100...⟩ + |010...⟩ + ... + |...001⟩)/√n
        
        W state is more robust to qubit loss than GHZ.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            W state circuit
        """
        n = num_qubits
        circuit = QuantumCircuit(n, n, name=f"W_{n}")
        
        # Initialize first qubit to |1⟩
        circuit.x(0)
        
        # Distribute the |1⟩ across all qubits
        for i in range(n - 1):
            # Controlled rotation to distribute amplitude
            angle = 2 * np.arccos(np.sqrt(1 / (n - i)))
            circuit.cry(angle, i, i + 1)
            circuit.cx(i + 1, i)
        
        return circuit
