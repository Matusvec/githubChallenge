"""
Quantum Simulator Engine
========================

Core simulation engine supporting:
- Unitary evolution
- Noisy quantum channels
- Time-dependent Hamiltonians
- Circuit execution with intermediate measurements
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from quantumviz.core.quantum_state import QuantumState
from quantumviz.core.gates import QuantumGates, Gate
from quantumviz.core.noise import NoiseModel, NoiseChannel


class SimulationMode(Enum):
    """Simulation modes."""
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    MONTE_CARLO = "monte_carlo"


@dataclass
class CircuitInstruction:
    """Represents a single instruction in a quantum circuit."""
    gate_name: str
    target_qubits: List[int]
    params: Optional[Dict[str, float]] = None
    condition: Optional[Tuple[int, int]] = None  # (classical_bit, value)
    
    def __repr__(self) -> str:
        return f"{self.gate_name}({self.target_qubits})"


@dataclass
class CircuitLayer:
    """A layer of gates that can be executed in parallel."""
    instructions: List[CircuitInstruction] = field(default_factory=list)
    
    def add(self, instruction: CircuitInstruction):
        self.instructions.append(instruction)


@dataclass
class SimulationResult:
    """Results from a quantum simulation."""
    final_state: QuantumState
    counts: Optional[Dict[str, int]] = None
    state_history: List[QuantumState] = field(default_factory=list)
    execution_time: float = 0.0
    gate_count: int = 0
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities."""
        probs = self.final_state.get_probabilities()
        n = self.final_state.num_qubits
        return {format(i, f'0{n}b'): float(p) for i, p in enumerate(probs) if p > 1e-10}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_state": self.final_state.to_dict(),
            "counts": self.counts,
            "execution_time": self.execution_time,
            "gate_count": self.gate_count,
            "depth": self.depth,
            "probabilities": self.get_probabilities(),
        }


class QuantumCircuit:
    """
    Quantum circuit representation with layered structure for visualization.
    """
    
    def __init__(self, num_qubits: int, num_classical_bits: int = 0, name: str = "circuit"):
        """
        Initialize quantum circuit.
        
        Args:
            num_qubits: Number of qubits
            num_classical_bits: Number of classical bits for measurement
            name: Circuit identifier
        """
        self.num_qubits = num_qubits
        self.num_classical_bits = num_classical_bits or num_qubits
        self.name = name
        self.instructions: List[CircuitInstruction] = []
        self.layers: List[CircuitLayer] = []
        self._current_layer: Optional[CircuitLayer] = None
        self._qubit_last_used: Dict[int, int] = {}  # Track layer for each qubit
    
    def _add_instruction(self, gate_name: str, targets: List[int], 
                         params: Optional[Dict[str, float]] = None,
                         condition: Optional[Tuple[int, int]] = None):
        """Internal method to add instruction and manage layers."""
        instruction = CircuitInstruction(gate_name, targets, params, condition)
        self.instructions.append(instruction)
        
        # Determine which layer this gate belongs to
        max_layer = max((self._qubit_last_used.get(q, -1) for q in targets), default=-1)
        target_layer = max_layer + 1
        
        # Ensure we have enough layers
        while len(self.layers) <= target_layer:
            self.layers.append(CircuitLayer())
        
        self.layers[target_layer].add(instruction)
        
        # Update qubit usage
        for q in targets:
            self._qubit_last_used[q] = target_layer
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SINGLE-QUBIT GATES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def h(self, qubit: int) -> "QuantumCircuit":
        """Add Hadamard gate."""
        self._add_instruction("H", [qubit])
        return self
    
    def x(self, qubit: int) -> "QuantumCircuit":
        """Add Pauli-X gate."""
        self._add_instruction("X", [qubit])
        return self
    
    def y(self, qubit: int) -> "QuantumCircuit":
        """Add Pauli-Y gate."""
        self._add_instruction("Y", [qubit])
        return self
    
    def z(self, qubit: int) -> "QuantumCircuit":
        """Add Pauli-Z gate."""
        self._add_instruction("Z", [qubit])
        return self
    
    def s(self, qubit: int) -> "QuantumCircuit":
        """Add S gate."""
        self._add_instruction("S", [qubit])
        return self
    
    def t(self, qubit: int) -> "QuantumCircuit":
        """Add T gate."""
        self._add_instruction("T", [qubit])
        return self
    
    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Add RX rotation gate."""
        self._add_instruction("RX", [qubit], {"theta": theta})
        return self
    
    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Add RY rotation gate."""
        self._add_instruction("RY", [qubit], {"theta": theta})
        return self
    
    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Add RZ rotation gate."""
        self._add_instruction("RZ", [qubit], {"theta": theta})
        return self
    
    def p(self, phi: float, qubit: int) -> "QuantumCircuit":
        """Add phase gate."""
        self._add_instruction("P", [qubit], {"phi": phi})
        return self
    
    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> "QuantumCircuit":
        """Add U3 gate."""
        self._add_instruction("U3", [qubit], {"theta": theta, "phi": phi, "lambda": lam})
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TWO-QUBIT GATES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Add CNOT (CX) gate."""
        self._add_instruction("CNOT", [control, target])
        return self
    
    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """Alias for cx."""
        return self.cx(control, target)
    
    def cy(self, control: int, target: int) -> "QuantumCircuit":
        """Add CY gate."""
        self._add_instruction("CY", [control, target])
        return self
    
    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Add CZ gate."""
        self._add_instruction("CZ", [control, target])
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> "QuantumCircuit":
        """Add SWAP gate."""
        self._add_instruction("SWAP", [qubit1, qubit2])
        return self
    
    def crx(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        """Add controlled RX gate."""
        self._add_instruction("CRX", [control, target], {"theta": theta})
        return self
    
    def cry(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        """Add controlled RY gate."""
        self._add_instruction("CRY", [control, target], {"theta": theta})
        return self
    
    def crz(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        """Add controlled RZ gate."""
        self._add_instruction("CRZ", [control, target], {"theta": theta})
        return self
    
    def cp(self, phi: float, control: int, target: int) -> "QuantumCircuit":
        """Add controlled phase gate."""
        self._add_instruction("CP", [control, target], {"phi": phi})
        return self
    
    def xx(self, theta: float, qubit1: int, qubit2: int) -> "QuantumCircuit":
        """Add XX Ising gate."""
        self._add_instruction("XX", [qubit1, qubit2], {"theta": theta})
        return self
    
    def yy(self, theta: float, qubit1: int, qubit2: int) -> "QuantumCircuit":
        """Add YY Ising gate."""
        self._add_instruction("YY", [qubit1, qubit2], {"theta": theta})
        return self
    
    def zz(self, theta: float, qubit1: int, qubit2: int) -> "QuantumCircuit":
        """Add ZZ Ising gate."""
        self._add_instruction("ZZ", [qubit1, qubit2], {"theta": theta})
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-QUBIT GATES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def ccx(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        """Add Toffoli (CCX) gate."""
        self._add_instruction("Toffoli", [control1, control2, target])
        return self
    
    def toffoli(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        """Alias for ccx."""
        return self.ccx(control1, control2, target)
    
    def cswap(self, control: int, target1: int, target2: int) -> "QuantumCircuit":
        """Add Fredkin (CSWAP) gate."""
        self._add_instruction("Fredkin", [control, target1, target2])
        return self
    
    def fredkin(self, control: int, target1: int, target2: int) -> "QuantumCircuit":
        """Alias for cswap."""
        return self.cswap(control, target1, target2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MEASUREMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def measure(self, qubit: int, classical_bit: int) -> "QuantumCircuit":
        """Add measurement."""
        self._add_instruction("MEASURE", [qubit], {"classical_bit": classical_bit})
        return self
    
    def measure_all(self) -> "QuantumCircuit":
        """Measure all qubits to corresponding classical bits."""
        for i in range(self.num_qubits):
            self.measure(i, i)
        return self
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BARRIERS AND UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def barrier(self, *qubits) -> "QuantumCircuit":
        """Add barrier (for visualization and layer separation)."""
        targets = list(qubits) if qubits else list(range(self.num_qubits))
        self._add_instruction("BARRIER", targets)
        return self
    
    def reset(self, qubit: int) -> "QuantumCircuit":
        """Reset qubit to |0⟩."""
        self._add_instruction("RESET", [qubit])
        return self
    
    @property
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        return len(self.layers)
    
    @property
    def gate_count(self) -> int:
        """Total number of gates."""
        return len([i for i in self.instructions if i.gate_name not in ["BARRIER", "MEASURE"]])
    
    def copy(self) -> "QuantumCircuit":
        """Create a copy of the circuit."""
        new_circuit = QuantumCircuit(self.num_qubits, self.num_classical_bits, self.name)
        for instr in self.instructions:
            new_circuit._add_instruction(
                instr.gate_name, 
                instr.target_qubits.copy(),
                instr.params.copy() if instr.params else None,
                instr.condition
            )
        return new_circuit
    
    def __repr__(self) -> str:
        return f"QuantumCircuit(qubits={self.num_qubits}, depth={self.depth}, gates={self.gate_count})"
    
    def draw_ascii(self) -> str:
        """Generate ASCII representation of the circuit."""
        lines = [f"q{i}: " for i in range(self.num_qubits)]
        
        for layer in self.layers:
            max_len = max(len(line) for line in lines)
            lines = [line.ljust(max_len) for line in lines]
            
            for instr in layer.instructions:
                if instr.gate_name == "BARRIER":
                    for q in instr.target_qubits:
                        lines[q] += "│"
                elif instr.gate_name == "MEASURE":
                    q = instr.target_qubits[0]
                    lines[q] += "─M─"
                elif len(instr.target_qubits) == 1:
                    q = instr.target_qubits[0]
                    name = instr.gate_name[:3]
                    lines[q] += f"─[{name}]─"
                elif len(instr.target_qubits) == 2:
                    q1, q2 = instr.target_qubits
                    if instr.gate_name == "CNOT":
                        lines[q1] += "──●──"
                        lines[q2] += "──⊕──"
                    elif instr.gate_name == "CZ":
                        lines[q1] += "──●──"
                        lines[q2] += "──●──"
                    elif instr.gate_name == "SWAP":
                        lines[q1] += "──×──"
                        lines[q2] += "──×──"
                    else:
                        name = instr.gate_name[:2]
                        lines[q1] += f"─[{name}]─"
                        lines[q2] += f"─[{name}]─"
        
        return "\n".join(lines)


class QuantumSimulator:
    """
    High-performance quantum circuit simulator.
    
    Supports:
    - Statevector and density matrix simulation
    - Noisy simulation with custom noise models
    - Circuit optimization
    - Intermediate state capture for visualization
    """
    
    def __init__(self, mode: SimulationMode = SimulationMode.DENSITY_MATRIX):
        """
        Initialize simulator.
        
        Args:
            mode: Simulation mode (statevector, density_matrix, or monte_carlo)
        """
        self.mode = mode
        self.noise_model: Optional[NoiseModel] = None
        self.capture_history: bool = False
        self.optimization_level: int = 0
    
    def set_noise_model(self, noise_model: NoiseModel):
        """Set noise model for noisy simulation."""
        self.noise_model = noise_model
        if noise_model is not None:
            self.mode = SimulationMode.DENSITY_MATRIX
    
    def enable_history_capture(self, enable: bool = True):
        """Enable capturing intermediate states for visualization."""
        self.capture_history = enable
    
    def _get_gate(self, instruction: CircuitInstruction) -> Gate:
        """Convert instruction to gate object."""
        params = instruction.params or {}
        return QuantumGates.get_gate_by_name(instruction.gate_name, **params)
    
    def _apply_gate(self, state: QuantumState, instruction: CircuitInstruction):
        """Apply a gate instruction to the state."""
        if instruction.gate_name in ["BARRIER", "MEASURE", "RESET"]:
            return
        
        gate = self._get_gate(instruction)
        expanded_matrix = QuantumGates.expand_gate(
            gate, 
            instruction.target_qubits, 
            state.num_qubits
        )
        
        state.apply_operator(expanded_matrix, operation_name=instruction.gate_name)
        
        # Apply noise if configured
        if self.noise_model is not None:
            noise_channel = self.noise_model.get_noise_for_gate(instruction.gate_name)
            if noise_channel is not None:
                # Expand noise to full system (simplified for single-qubit noise)
                for target in instruction.target_qubits:
                    self._apply_noise_to_qubit(state, noise_channel, target)
    
    def _apply_noise_to_qubit(self, state: QuantumState, channel: NoiseChannel, qubit: int):
        """Apply noise channel to a specific qubit in the system."""
        n = state.num_qubits
        dim = 2 ** n
        
        if state.density_matrix is None:
            return
        
        # For simplicity, apply noise in computational basis
        # This is an approximation for multi-qubit systems
        for K in channel.kraus_operators:
            K_expanded = QuantumGates._expand_single_qubit(K, qubit, n)
            state.density_matrix = K_expanded @ state.density_matrix @ np.conj(K_expanded.T)
    
    def run(self, 
            circuit: QuantumCircuit, 
            initial_state: Optional[QuantumState] = None,
            shots: int = 0) -> SimulationResult:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: Circuit to execute
            initial_state: Optional initial state (default: |0...0⟩)
            shots: Number of measurement shots (0 for statevector only)
            
        Returns:
            SimulationResult with final state and optional counts
        """
        start_time = time.time()
        
        # Initialize state
        if initial_state is not None:
            state = QuantumState(
                num_qubits=circuit.num_qubits,
                state_vector=initial_state.state_vector.copy() if initial_state.state_vector is not None else None,
                density_matrix=initial_state.density_matrix.copy() if initial_state.density_matrix is not None else None,
                name=f"{circuit.name}_result"
            )
        else:
            state = QuantumState(num_qubits=circuit.num_qubits, name=f"{circuit.name}_result")
        
        state_history = []
        if self.capture_history:
            state_history.append(self._copy_state(state))
        
        # Execute instructions
        for instruction in circuit.instructions:
            if instruction.gate_name not in ["BARRIER", "MEASURE", "RESET"]:
                self._apply_gate(state, instruction)
                
                if self.capture_history:
                    state_history.append(self._copy_state(state))
        
        # Perform measurements if requested
        counts = None
        if shots > 0:
            counts = state.measure(shots)
            if self.noise_model is not None:
                counts = self.noise_model.apply_measurement_error(counts)
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            final_state=state,
            counts=counts,
            state_history=state_history,
            execution_time=execution_time,
            gate_count=circuit.gate_count,
            depth=circuit.depth,
            metadata={
                "circuit_name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "noise_model": self.noise_model.to_dict() if self.noise_model else None,
            }
        )
    
    def _copy_state(self, state: QuantumState) -> QuantumState:
        """Create a copy of a quantum state."""
        return QuantumState(
            num_qubits=state.num_qubits,
            state_vector=state.state_vector.copy() if state.state_vector is not None else None,
            density_matrix=state.density_matrix.copy() if state.density_matrix is not None else None,
            name=state.name
        )
    
    def time_evolution(self, 
                       hamiltonian: np.ndarray, 
                       initial_state: QuantumState,
                       times: np.ndarray,
                       num_qubits: int) -> List[QuantumState]:
        """
        Simulate time evolution under a Hamiltonian.
        
        |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        
        Args:
            hamiltonian: Hamiltonian matrix (Hermitian)
            initial_state: Initial quantum state
            times: Array of time points
            num_qubits: Number of qubits
            
        Returns:
            List of states at each time point
        """
        from scipy.linalg import expm
        
        states = []
        
        for t in times:
            # Unitary time evolution operator
            U = expm(-1j * hamiltonian * t)
            
            state = QuantumState(
                num_qubits=num_qubits,
                state_vector=initial_state.state_vector.copy() if initial_state.state_vector is not None else None,
                density_matrix=initial_state.density_matrix.copy() if initial_state.density_matrix is not None else None,
                name=f"t={t:.4f}"
            )
            
            state.apply_operator(U, operation_name=f"U(t={t:.4f})")
            states.append(state)
        
        return states
    
    def simulate_decoherence(self, 
                             initial_state: QuantumState,
                             t1: float, 
                             t2: float,
                             times: np.ndarray) -> List[QuantumState]:
        """
        Simulate T1/T2 decoherence over time.
        
        Args:
            initial_state: Initial quantum state
            t1: T1 relaxation time
            t2: T2 dephasing time
            times: Array of time points
            
        Returns:
            List of states showing decoherence evolution
        """
        states = []
        
        for t in times:
            state = self._copy_state(initial_state)
            state.name = f"decoherence_t={t:.4f}"
            
            # Apply thermal relaxation to each qubit
            channel = NoiseModel.thermal_relaxation_channel(t1, t2, t)
            
            for qubit in range(state.num_qubits):
                self._apply_noise_to_qubit(state, channel, qubit)
            
            states.append(state)
        
        return states
    
    def get_expectation_value(self, state: QuantumState, observable: np.ndarray) -> complex:
        """
        Compute expectation value of an observable.
        
        ⟨O⟩ = Tr(ρO)
        
        Args:
            state: Quantum state
            observable: Observable operator (Hermitian matrix)
            
        Returns:
            Expectation value
        """
        if state.density_matrix is not None:
            return np.trace(state.density_matrix @ observable)
        elif state.state_vector is not None:
            return np.conj(state.state_vector) @ observable @ state.state_vector
        return 0.0
    
    def __repr__(self) -> str:
        noise = "noisy" if self.noise_model else "ideal"
        return f"QuantumSimulator(mode={self.mode.value}, {noise})"
