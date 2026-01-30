"""
Natural Language Parser for Quantum Circuits
=============================================

Pattern-based parser for converting natural language descriptions
into quantum circuits without requiring the Copilot SDK.

This serves as a fallback when the GitHub Copilot SDK is not available.
"""

from __future__ import annotations
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedCommand:
    """Represents a parsed natural language command."""
    circuit_type: str
    num_qubits: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    modifications: List[str] = field(default_factory=list)
    confidence: float = 0.5


class NaturalLanguageParser:
    """
    Parse natural language descriptions into quantum circuits.
    
    This parser uses rule-based pattern matching to convert
    natural language into quantum circuit specifications.
    
    Example:
        >>> parser = NaturalLanguageParser()
        >>> circuit = parser.parse("Create a 3-qubit GHZ state with noise")
    """
    
    # Circuit type patterns
    CIRCUIT_PATTERNS = [
        # Bell states
        (r"bell\s*state|bell\s*pair|epr\s*pair", "bell", {"num_qubits": 2}),
        (r"(?:create\s+)?entangle(?:d|ment)", "bell", {"num_qubits": 2}),
        
        # GHZ states
        (r"ghz\s*state", "ghz", {}),
        (r"greenberger|cat\s*state", "ghz", {}),
        (r"(\d+)\s*-?\s*(?:qubit\s+)?ghz", "ghz", {}),
        
        # W states
        (r"w\s*state|dicke\s*state", "w_state", {}),
        
        # Grover's algorithm
        (r"grover|amplitude\s*amplif", "grover", {}),
        (r"search\s*(?:for|algorithm)", "grover", {}),
        (r"oracle\s*search", "grover", {}),
        
        # QFT
        (r"qft|quantum\s*fourier", "qft", {}),
        (r"inverse\s*(?:qft|fourier)", "qft", {"inverse": True}),
        (r"iqft", "qft", {"inverse": True}),
        
        # VQE
        (r"vqe|variational\s*(?:quantum\s*)?eigen", "vqe", {}),
        (r"ground\s*state|hamiltonian", "vqe", {}),
        
        # Phase estimation
        (r"qpe|phase\s*estim", "qpe", {}),
        
        # Teleportation
        (r"teleport", "teleport", {"num_qubits": 3}),
        
        # Random circuit
        (r"random\s*circuit", "random", {}),
        
        # Basic states
        (r"superposition", "superposition", {}),
        (r"\|?\+\>", "plus", {"num_qubits": 1}),
        (r"\|-\>", "minus", {"num_qubits": 1}),
    ]
    
    # Modifier patterns
    MODIFIER_PATTERNS = [
        (r"with\s*noise|noisy", "noise"),
        (r"with\s*measurement|measure", "measure"),
        (r"inverse|adjoint|dagger", "inverse"),
        (r"controlled", "controlled"),
        (r"repeat(?:ed)?\s*(\d+)", "repeat"),
    ]
    
    def __init__(self):
        """Initialize the parser."""
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), circuit_type, defaults)
            for pattern, circuit_type, defaults in self.CIRCUIT_PATTERNS
        ]
        
        self._modifier_patterns = [
            (re.compile(pattern, re.IGNORECASE), modifier)
            for pattern, modifier in self.MODIFIER_PATTERNS
        ]
    
    def parse(
        self,
        text: str,
        default_qubits: int = 2,
    ) -> Any:
        """
        Parse natural language text into a quantum circuit.
        
        Args:
            text: Natural language description
            default_qubits: Default number of qubits if not specified
            
        Returns:
            QuantumCircuit object
        """
        # Parse the command
        command = self._parse_command(text, default_qubits)
        
        # Build the circuit
        circuit = self._build_circuit(command)
        
        return circuit
    
    def _parse_command(self, text: str, default_qubits: int) -> ParsedCommand:
        """Parse text into a ParsedCommand."""
        text_lower = text.lower().strip()
        
        # Detect circuit type
        circuit_type = "superposition"
        params = {}
        confidence = 0.3
        
        for pattern, ctype, defaults in self._compiled_patterns:
            if pattern.search(text_lower):
                circuit_type = ctype
                params.update(defaults)
                confidence = 0.7
                break
        
        # Extract number of qubits
        num_qubits = self._extract_qubits(text_lower, params.get("num_qubits", default_qubits))
        
        # Extract additional parameters
        params.update(self._extract_parameters(text_lower))
        
        # Detect modifiers
        modifications = []
        for pattern, modifier in self._modifier_patterns:
            match = pattern.search(text_lower)
            if match:
                modifications.append(modifier)
                if modifier == "repeat" and match.groups():
                    params["repeat_count"] = int(match.group(1))
        
        return ParsedCommand(
            circuit_type=circuit_type,
            num_qubits=num_qubits,
            parameters=params,
            modifications=modifications,
            confidence=confidence,
        )
    
    def _extract_qubits(self, text: str, default: int) -> int:
        """Extract number of qubits from text."""
        patterns = [
            r"(\d+)\s*-?\s*qubit",
            r"(\d+)\s*qubits",
            r"n\s*=\s*(\d+)",
            r"(\d+)\s*-?\s*(?:qubit\s+)?(?:ghz|bell|w\s*state)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return default
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract additional parameters from text."""
        params = {}
        
        # Target state for Grover
        match = re.search(r"(?:target|find|search\s*for)\s*(?:state\s*)?(\d+)", text)
        if match:
            params["target"] = int(match.group(1))
        
        # Multiple targets
        match = re.search(r"targets?\s*[\[\(]?([\d,\s]+)[\]\)]?", text)
        if match:
            targets = [int(x.strip()) for x in match.group(1).split(",")]
            params["targets"] = targets
        
        # Number of iterations
        match = re.search(r"(\d+)\s*iteration", text)
        if match:
            params["iterations"] = int(match.group(1))
        
        # Layers for VQE/variational
        match = re.search(r"(\d+)\s*layer", text)
        if match:
            params["layers"] = int(match.group(1))
        
        # Depth
        match = re.search(r"depth\s*(?:of\s*)?(\d+)", text)
        if match:
            params["depth"] = int(match.group(1))
        
        # Angle parameters
        match = re.search(r"angle\s*(?:of\s*)?([\d.]+)", text)
        if match:
            params["angle"] = float(match.group(1))
        
        # Bell state type
        for bell_type in ["phi+", "phi-", "psi+", "psi-"]:
            if bell_type in text.lower().replace("ψ", "psi").replace("φ", "phi"):
                params["bell_type"] = bell_type
                break
        
        # Noise level
        match = re.search(r"(low|moderate|medium|high)\s*noise", text)
        if match:
            params["noise_level"] = match.group(1)
        
        return params
    
    def _build_circuit(self, command: ParsedCommand) -> Any:
        """Build a quantum circuit from parsed command."""
        from quantumviz.core.simulator import QuantumCircuit
        from quantumviz.core.algorithms import QuantumAlgorithms
        
        circuit_type = command.circuit_type
        num_qubits = command.num_qubits
        params = command.parameters
        
        # Build circuit based on type
        if circuit_type == "bell":
            circuit = QuantumAlgorithms.bell_state_circuit(
                params.get("bell_type", "phi+")
            )
            
        elif circuit_type == "ghz":
            circuit = QuantumAlgorithms.ghz_state_circuit(num_qubits)
            
        elif circuit_type == "w_state":
            circuit = QuantumAlgorithms.w_state_circuit(num_qubits)
            
        elif circuit_type == "grover":
            targets = params.get("targets", params.get("target", [1]))
            if isinstance(targets, int):
                targets = [targets]
            circuit, _ = QuantumAlgorithms.grover_search(
                num_qubits,
                targets,
                iterations=params.get("iterations"),
            )
            
        elif circuit_type == "qft":
            circuit = QuantumAlgorithms.qft(
                num_qubits,
                inverse=params.get("inverse", False),
            )
            
        elif circuit_type == "qpe":
            circuit = QuantumAlgorithms.quantum_phase_estimation(
                precision_qubits=num_qubits - 1
            )
            
        elif circuit_type == "vqe":
            # Create a simple variational ansatz
            import numpy as np
            circuit = QuantumCircuit(num_qubits, name="VQE_Ansatz")
            layers = params.get("layers", 2)
            
            for layer in range(layers):
                for q in range(num_qubits):
                    circuit.ry(q, np.pi / 4 * (layer + 1))
                    circuit.rz(q, np.pi / 4 * (layer + 1))
                for q in range(num_qubits - 1):
                    circuit.cnot(q, q + 1)
                    
        elif circuit_type == "teleport":
            circuit = QuantumCircuit(3, name="Teleportation")
            # Create Bell pair
            circuit.h(1)
            circuit.cnot(1, 2)
            # Bell measurement
            circuit.cnot(0, 1)
            circuit.h(0)
            
        elif circuit_type == "random":
            import numpy as np
            circuit = QuantumCircuit(num_qubits, name="Random")
            depth = params.get("depth", 5)
            
            for _ in range(depth):
                gate = np.random.choice(["h", "x", "y", "z", "s", "t"])
                qubit = np.random.randint(0, num_qubits)
                getattr(circuit, gate)(qubit)
                
                if num_qubits > 1 and np.random.random() > 0.5:
                    q1, q2 = np.random.choice(num_qubits, 2, replace=False)
                    circuit.cnot(int(q1), int(q2))
                    
        elif circuit_type == "plus":
            circuit = QuantumCircuit(1, name="Plus")
            circuit.h(0)
            
        elif circuit_type == "minus":
            circuit = QuantumCircuit(1, name="Minus")
            circuit.x(0)
            circuit.h(0)
            
        else:  # superposition
            circuit = QuantumCircuit(num_qubits, name="Superposition")
            for q in range(num_qubits):
                circuit.h(q)
        
        # Apply modifiers
        if "measure" in command.modifications:
            for q in range(circuit.num_qubits):
                circuit.measure(q)
        
        if "repeat" in command.modifications:
            repeat_count = params.get("repeat_count", 2)
            original_ops = circuit.operations.copy()
            for _ in range(repeat_count - 1):
                for op in original_ops:
                    circuit.operations.append(op)
        
        return circuit
    
    def suggest_completions(self, partial_text: str) -> List[str]:
        """Suggest completions for partial input."""
        suggestions = []
        
        partial_lower = partial_text.lower()
        
        # Suggest circuit types
        circuit_suggestions = [
            "Bell state",
            "GHZ state",
            "W state",
            "Grover search",
            "QFT (Quantum Fourier Transform)",
            "VQE circuit",
            "Phase estimation",
            "Superposition",
            "Random circuit",
        ]
        
        for suggestion in circuit_suggestions:
            if suggestion.lower().startswith(partial_lower) or any(
                word.startswith(partial_lower) for word in suggestion.lower().split()
            ):
                suggestions.append(suggestion)
        
        # Add qubit count suggestions
        if re.search(r"\d+$", partial_text):
            suggestions.extend([
                f"{partial_text}-qubit circuit",
                f"{partial_text} qubits",
            ])
        
        return suggestions[:5]  # Return top 5
    
    def explain_circuit(self, circuit_type: str) -> str:
        """Get explanation for a circuit type."""
        explanations = {
            "bell": "Bell state: Maximally entangled 2-qubit state |Φ+⟩ = (|00⟩ + |11⟩)/√2",
            "ghz": "GHZ state: Multi-qubit entanglement (|00...0⟩ + |11...1⟩)/√2",
            "w_state": "W state: Symmetric entangled state with exactly one qubit in |1⟩",
            "grover": "Grover's algorithm: Quantum search with quadratic speedup",
            "qft": "Quantum Fourier Transform: Basis for phase estimation and factoring",
            "vqe": "VQE: Variational algorithm for finding ground state energies",
            "qpe": "Phase estimation: Determines eigenvalues of unitary operators",
            "teleport": "Quantum teleportation: Transfer quantum state using entanglement",
            "superposition": "Equal superposition of all computational basis states",
        }
        
        return explanations.get(
            circuit_type,
            f"Unknown circuit type: {circuit_type}"
        )
