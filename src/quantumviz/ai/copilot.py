"""
GitHub Copilot SDK Integration
==============================

🤖 AI-powered quantum circuit generation using GitHub Copilot SDK.

This module leverages the GitHub Copilot SDK to enable natural language
circuit generation, intelligent suggestions, and code completion for
quantum computing workflows.

Usage:
    from quantumviz.ai.copilot import CopilotCircuitGenerator
    
    generator = CopilotCircuitGenerator()
    circuit, description = generator.generate_circuit("3-qubit GHZ state")
"""

from __future__ import annotations
import os
import re
import json
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from AI circuit generation."""
    circuit: Any  # QuantumCircuit
    description: str
    confidence: float
    suggestions: List[str] = field(default_factory=list)
    code: Optional[str] = None
    tokens_used: int = 0


class CopilotCircuitGenerator:
    """
    AI-powered quantum circuit generator using GitHub Copilot SDK.
    
    This class provides an interface to generate quantum circuits from
    natural language descriptions using the GitHub Copilot SDK.
    
    Features:
    - Natural language to circuit conversion
    - Intelligent parameter suggestions
    - Circuit optimization hints
    - Error correction recommendations
    
    Example:
        >>> generator = CopilotCircuitGenerator()
        >>> circuit, desc = generator.generate_circuit("Create a Bell state")
        >>> print(circuit)
    """
    
    # Supported circuit patterns
    CIRCUIT_PATTERNS = {
        "bell": r"bell\s*state|entangl(ed|ement)|epr\s*pair",
        "ghz": r"ghz|greenberger|cat\s*state|multi.*qubit.*entangl",
        "grover": r"grover|search|amplitude\s*amplif|oracle",
        "qft": r"qft|fourier|quantum\s*fourier",
        "vqe": r"vqe|variational|eigensolv|ground\s*state|hamiltonian",
        "qpe": r"qpe|phase\s*estim",
        "w_state": r"w\s*state|dicke",
        "random": r"random|arbitrary",
        "teleport": r"teleport",
        "swap": r"swap\s*test|swap\s*circuit",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.3,
        use_cache: bool = True,
    ):
        """
        Initialize the Copilot Circuit Generator.
        
        Args:
            api_key: GitHub Copilot API key (or use COPILOT_API_KEY env var)
            model: Model to use for generation
            temperature: Sampling temperature (lower = more deterministic)
            use_cache: Whether to cache generation results
        """
        self.api_key = api_key or os.environ.get("COPILOT_API_KEY") or os.environ.get("GITHUB_TOKEN")
        self.model = model
        self.temperature = temperature
        self.use_cache = use_cache
        self._cache: Dict[str, GenerationResult] = {}
        
        # Try to import GitHub Copilot SDK
        self._copilot_available = False
        try:
            import github_copilot_sdk
            self._copilot = github_copilot_sdk
            self._copilot_available = True
            logger.info("GitHub Copilot SDK loaded successfully")
        except ImportError:
            logger.warning(
                "GitHub Copilot SDK not available. "
                "Using fallback pattern-based parsing. "
                "Install with: pip install github-copilot-sdk"
            )
    
    @property
    def is_copilot_available(self) -> bool:
        """Check if GitHub Copilot SDK is available."""
        return self._copilot_available
    
    def generate_circuit(
        self,
        prompt: str,
        num_qubits: Optional[int] = None,
        with_measurements: bool = True,
    ) -> Tuple[Any, str]:
        """
        Generate a quantum circuit from a natural language prompt.
        
        Args:
            prompt: Natural language description of the circuit
            num_qubits: Optional number of qubits (auto-detected if not provided)
            with_measurements: Whether to add measurements at the end
            
        Returns:
            Tuple of (QuantumCircuit, description string)
            
        Example:
            >>> circuit, desc = generator.generate_circuit("3-qubit GHZ state")
        """
        # Check cache
        cache_key = f"{prompt}:{num_qubits}:{with_measurements}"
        if self.use_cache and cache_key in self._cache:
            result = self._cache[cache_key]
            return result.circuit, result.description
        
        # Normalize prompt
        prompt_lower = prompt.lower().strip()
        
        # Extract qubit count from prompt if not provided
        if num_qubits is None:
            num_qubits = self._extract_qubit_count(prompt_lower)
        
        # Try Copilot SDK first
        if self._copilot_available:
            try:
                result = self._generate_with_copilot(prompt, num_qubits, with_measurements)
                if self.use_cache:
                    self._cache[cache_key] = result
                return result.circuit, result.description
            except Exception as e:
                logger.warning(f"Copilot generation failed: {e}. Using fallback.")
        
        # Fallback to pattern-based parsing
        result = self._generate_with_patterns(prompt_lower, num_qubits, with_measurements)
        if self.use_cache:
            self._cache[cache_key] = result
        
        return result.circuit, result.description
    
    def _generate_with_copilot(
        self,
        prompt: str,
        num_qubits: int,
        with_measurements: bool,
    ) -> GenerationResult:
        """Generate circuit using GitHub Copilot SDK."""
        # Build the generation prompt
        system_prompt = """You are a quantum computing expert. Generate quantum circuits using the following format.
        
Available gates: H, X, Y, Z, S, T, RX(theta), RY(theta), RZ(theta), CNOT, CZ, SWAP, CCX (Toffoli), CSWAP (Fredkin)

Respond with a JSON object containing:
- gates: List of gates to apply, each with {name, qubits, params (optional)}
- description: Brief description of what the circuit does
- optimal_shots: Recommended number of measurement shots

Example for Bell state:
{
    "gates": [
        {"name": "H", "qubits": [0]},
        {"name": "CNOT", "qubits": [0, 1]}
    ],
    "description": "Creates maximally entangled Bell state (|00⟩ + |11⟩)/√2",
    "optimal_shots": 1000
}"""
        
        user_prompt = f"""Create a quantum circuit for: {prompt}
        
Number of qubits: {num_qubits}
Add measurements: {with_measurements}

Respond only with valid JSON."""
        
        # Call Copilot API
        try:
            response = self._copilot.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
        except Exception as e:
            logger.error(f"Copilot API error: {e}")
            raise
        
        # Parse the response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                circuit_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Copilot response: {e}")
            logger.debug(f"Response was: {content}")
            raise
        
        # Build circuit from parsed data
        circuit = self._build_circuit_from_data(
            circuit_data, 
            num_qubits, 
            with_measurements
        )
        
        return GenerationResult(
            circuit=circuit,
            description=circuit_data.get("description", f"Circuit generated from: {prompt}"),
            confidence=0.9,
            code=content,
            tokens_used=tokens_used,
        )
    
    def _generate_with_patterns(
        self,
        prompt: str,
        num_qubits: int,
        with_measurements: bool,
    ) -> GenerationResult:
        """Fallback pattern-based circuit generation."""
        from quantumviz.core.simulator import QuantumCircuit
        from quantumviz.core.algorithms import QuantumAlgorithms
        
        # Detect circuit type from patterns
        circuit_type = self._detect_circuit_type(prompt)
        
        # Extract additional parameters
        params = self._extract_parameters(prompt)
        
        # Generate circuit based on type
        if circuit_type == "bell":
            circuit = QuantumAlgorithms.bell_state_circuit(params.get("bell_type", "phi+"))
            description = f"Bell state ({params.get('bell_type', 'Φ+')}) - maximally entangled 2-qubit state"
            
        elif circuit_type == "ghz":
            n = params.get("num_qubits", num_qubits)
            circuit = QuantumAlgorithms.ghz_state_circuit(n)
            description = f"GHZ state ({n} qubits) - (|{'0'*n}⟩ + |{'1'*n}⟩)/√2"
            
        elif circuit_type == "grover":
            targets = params.get("targets", [1])
            n = max(params.get("num_qubits", num_qubits), 2)
            circuit, _ = QuantumAlgorithms.grover_search(n, targets)
            description = f"Grover's search ({n} qubits) searching for states: {targets}"
            
        elif circuit_type == "qft":
            n = params.get("num_qubits", num_qubits)
            inverse = params.get("inverse", False)
            circuit = QuantumAlgorithms.qft(n, inverse=inverse)
            description = f"{'Inverse ' if inverse else ''}Quantum Fourier Transform ({n} qubits)"
            
        elif circuit_type == "w_state":
            n = params.get("num_qubits", num_qubits)
            circuit = QuantumAlgorithms.w_state_circuit(n)
            description = f"W state ({n} qubits) - symmetric entangled state"
            
        elif circuit_type == "qpe":
            n = params.get("precision", num_qubits - 1) or 3
            circuit = QuantumAlgorithms.quantum_phase_estimation(n)
            description = f"Quantum Phase Estimation ({n} precision qubits)"
            
        elif circuit_type == "vqe":
            n = params.get("num_qubits", num_qubits)
            layers = params.get("layers", 2)
            # Create a simple ansatz circuit
            circuit = QuantumCircuit(n, name="VQE_Ansatz")
            import numpy as np
            for layer in range(layers):
                for q in range(n):
                    circuit.ry(q, np.pi / 4)
                    circuit.rz(q, np.pi / 4)
                for q in range(n - 1):
                    circuit.cnot(q, q + 1)
            description = f"VQE Ansatz ({n} qubits, {layers} layers)"
            
        elif circuit_type == "teleport":
            circuit = QuantumCircuit(3, name="Quantum_Teleportation")
            # Prepare Bell pair (qubits 1-2)
            circuit.h(1)
            circuit.cnot(1, 2)
            # Bell measurement (qubits 0-1)
            circuit.cnot(0, 1)
            circuit.h(0)
            description = "Quantum Teleportation circuit (prepares Bell pair, needs classical corrections)"
            
        elif circuit_type == "random":
            import numpy as np
            circuit = QuantumCircuit(num_qubits, name="Random_Circuit")
            depth = params.get("depth", 5)
            gates = ["h", "x", "y", "z", "s", "t"]
            for _ in range(depth):
                for q in range(num_qubits):
                    gate = np.random.choice(gates)
                    getattr(circuit, gate)(q)
                if num_qubits > 1:
                    for q in range(num_qubits - 1):
                        if np.random.random() > 0.5:
                            circuit.cnot(q, q + 1)
            description = f"Random circuit ({num_qubits} qubits, depth {depth})"
            
        else:
            # Default: create superposition state
            circuit = QuantumCircuit(num_qubits, name="Superposition")
            for q in range(num_qubits):
                circuit.h(q)
            description = f"Superposition state ({num_qubits} qubits)"
        
        # Add measurements if requested
        if with_measurements:
            for q in range(circuit.num_qubits):
                circuit.measure(q)
        
        return GenerationResult(
            circuit=circuit,
            description=description,
            confidence=0.7,  # Lower confidence for pattern matching
            suggestions=self._get_suggestions(circuit_type, params),
        )
    
    def _build_circuit_from_data(
        self,
        data: Dict[str, Any],
        num_qubits: int,
        with_measurements: bool,
    ) -> Any:
        """Build a QuantumCircuit from parsed Copilot response data."""
        from quantumviz.core.simulator import QuantumCircuit
        
        gates = data.get("gates", [])
        
        # Determine number of qubits from gates
        max_qubit = max(
            (max(g.get("qubits", [0])) for g in gates),
            default=0
        )
        n_qubits = max(num_qubits, max_qubit + 1)
        
        circuit = QuantumCircuit(n_qubits, name=data.get("name", "AI_Generated"))
        
        # Apply gates
        for gate_info in gates:
            name = gate_info["name"].lower()
            qubits = gate_info.get("qubits", [0])
            params = gate_info.get("params", {})
            
            if name == "h":
                circuit.h(qubits[0])
            elif name == "x":
                circuit.x(qubits[0])
            elif name == "y":
                circuit.y(qubits[0])
            elif name == "z":
                circuit.z(qubits[0])
            elif name == "s":
                circuit.s(qubits[0])
            elif name == "t":
                circuit.t(qubits[0])
            elif name in ("rx", "r_x"):
                circuit.rx(qubits[0], params.get("theta", 0))
            elif name in ("ry", "r_y"):
                circuit.ry(qubits[0], params.get("theta", 0))
            elif name in ("rz", "r_z"):
                circuit.rz(qubits[0], params.get("theta", 0))
            elif name in ("cnot", "cx"):
                circuit.cnot(qubits[0], qubits[1])
            elif name == "cz":
                circuit.cz(qubits[0], qubits[1])
            elif name == "swap":
                circuit.swap(qubits[0], qubits[1])
            elif name in ("ccx", "toffoli", "ccnot"):
                circuit.ccx(qubits[0], qubits[1], qubits[2])
            elif name in ("cswap", "fredkin"):
                circuit.cswap(qubits[0], qubits[1], qubits[2])
        
        # Add measurements
        if with_measurements:
            for q in range(circuit.num_qubits):
                circuit.measure(q)
        
        return circuit
    
    def _detect_circuit_type(self, prompt: str) -> str:
        """Detect circuit type from prompt using patterns."""
        for circuit_type, pattern in self.CIRCUIT_PATTERNS.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return circuit_type
        return "superposition"
    
    def _extract_qubit_count(self, prompt: str) -> int:
        """Extract number of qubits from prompt."""
        # Look for explicit qubit count
        patterns = [
            r"(\d+)\s*-?\s*qubit",
            r"(\d+)\s*qubits",
            r"n\s*=\s*(\d+)",
            r"with\s*(\d+)",
            r"using\s*(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Default based on circuit type
        circuit_type = self._detect_circuit_type(prompt)
        defaults = {
            "bell": 2,
            "ghz": 3,
            "grover": 3,
            "qft": 3,
            "qpe": 4,
            "w_state": 3,
            "vqe": 2,
            "teleport": 3,
        }
        
        return defaults.get(circuit_type, 2)
    
    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract additional parameters from prompt."""
        params = {}
        
        # Extract qubit count
        match = re.search(r"(\d+)\s*-?\s*qubits?", prompt)
        if match:
            params["num_qubits"] = int(match.group(1))
        
        # Extract target states for Grover
        match = re.search(r"(?:target|state|find|search\s*for)\s*(\d+)", prompt)
        if match:
            params["targets"] = [int(match.group(1))]
        
        # Extract Bell state type
        for bell_type in ["phi+", "phi-", "psi+", "psi-", "Φ+", "Φ-", "Ψ+", "Ψ-"]:
            if bell_type.lower() in prompt:
                params["bell_type"] = bell_type.lower().replace("φ", "phi").replace("ψ", "psi")
                break
        
        # Extract iterations
        match = re.search(r"(\d+)\s*iteration", prompt)
        if match:
            params["iterations"] = int(match.group(1))
        
        # Extract layers for VQE
        match = re.search(r"(\d+)\s*layer", prompt)
        if match:
            params["layers"] = int(match.group(1))
        
        # Extract depth for random circuits
        match = re.search(r"depth\s*(\d+)", prompt)
        if match:
            params["depth"] = int(match.group(1))
        
        # Check for inverse QFT
        if "inverse" in prompt or "iqft" in prompt:
            params["inverse"] = True
        
        # Check for noise
        if "noisy" in prompt or "noise" in prompt:
            params["noise"] = True
            match = re.search(r"(low|moderate|high)\s*noise", prompt)
            if match:
                params["noise_level"] = match.group(1)
        
        return params
    
    def _get_suggestions(self, circuit_type: str, params: Dict[str, Any]) -> List[str]:
        """Get helpful suggestions based on circuit type."""
        suggestions = []
        
        if circuit_type == "grover":
            suggestions.append("For optimal success, number of iterations should be ~ π/4 * √N")
            if params.get("num_qubits", 0) > 4:
                suggestions.append("Consider using noise-adaptive strategies for larger circuits")
        
        elif circuit_type == "vqe":
            suggestions.append("Try different ansatz depths if convergence is slow")
            suggestions.append("Consider using ADAM optimizer for better convergence")
        
        elif circuit_type == "qft":
            suggestions.append("QFT is the basis for phase estimation and Shor's algorithm")
        
        elif circuit_type == "bell" or circuit_type == "ghz":
            suggestions.append("Use measure in different bases to verify entanglement")
        
        return suggestions
    
    def get_optimization_hints(self, circuit: Any) -> List[str]:
        """Get AI-powered optimization hints for a circuit."""
        hints = []
        
        gate_count = circuit.gate_count
        depth = circuit.depth
        num_qubits = circuit.num_qubits
        
        # Analyze circuit structure
        if depth > num_qubits * 3:
            hints.append("Circuit depth is high. Consider gate decomposition or parallelization.")
        
        if gate_count > 50:
            hints.append("Large circuit. Consider using approximate synthesis for optimization.")
        
        # Check for obvious optimizations
        ops = circuit.operations
        for i in range(len(ops) - 1):
            if ops[i][0] == ops[i+1][0] and ops[i][1] == ops[i+1][1]:
                if ops[i][0] in ["h", "x", "y", "z"]:
                    hints.append(f"Adjacent {ops[i][0].upper()} gates on qubit {ops[i][1]} can be removed (they cancel out)")
        
        return hints
    
    def generate_code(self, circuit: Any, framework: str = "quantumviz") -> str:
        """Generate code to recreate the circuit in various frameworks."""
        if framework == "quantumviz":
            return self._generate_quantumviz_code(circuit)
        elif framework == "qiskit":
            return self._generate_qiskit_code(circuit)
        elif framework == "cirq":
            return self._generate_cirq_code(circuit)
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
    def _generate_quantumviz_code(self, circuit: Any) -> str:
        """Generate QuantumViz code."""
        lines = [
            "from quantumviz.core.simulator import QuantumCircuit",
            "",
            f"circuit = QuantumCircuit({circuit.num_qubits}, name='{circuit.name}')",
            "",
        ]
        
        for op, *args in circuit.operations:
            if op == "measure":
                lines.append(f"circuit.measure({args[0]})")
            elif len(args) == 1:
                lines.append(f"circuit.{op}({args[0]})")
            elif len(args) == 2:
                if isinstance(args[1], float):
                    lines.append(f"circuit.{op}({args[0]}, {args[1]:.4f})")
                else:
                    lines.append(f"circuit.{op}({args[0]}, {args[1]})")
            else:
                lines.append(f"circuit.{op}({', '.join(map(str, args))})")
        
        return "\n".join(lines)
    
    def _generate_qiskit_code(self, circuit: Any) -> str:
        """Generate Qiskit code."""
        lines = [
            "from qiskit import QuantumCircuit",
            "",
            f"qc = QuantumCircuit({circuit.num_qubits}, {circuit.num_qubits})",
            "",
        ]
        
        gate_map = {
            "cnot": "cx",
            "ccx": "ccx",
            "cswap": "cswap",
        }
        
        for op, *args in circuit.operations:
            op_name = gate_map.get(op, op)
            if op == "measure":
                lines.append(f"qc.measure({args[0]}, {args[0]})")
            elif len(args) == 1:
                lines.append(f"qc.{op_name}({args[0]})")
            elif len(args) == 2:
                if isinstance(args[1], float):
                    lines.append(f"qc.{op_name}({args[1]:.4f}, {args[0]})")  # Qiskit: angle first
                else:
                    lines.append(f"qc.{op_name}({args[0]}, {args[1]})")
            else:
                lines.append(f"qc.{op_name}({', '.join(map(str, args))})")
        
        return "\n".join(lines)
    
    def _generate_cirq_code(self, circuit: Any) -> str:
        """Generate Cirq code."""
        lines = [
            "import cirq",
            "",
            f"qubits = cirq.LineQubit.range({circuit.num_qubits})",
            "circuit = cirq.Circuit()",
            "",
        ]
        
        gate_map = {
            "h": "H",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "s": "S",
            "t": "T",
            "cnot": "CNOT",
            "cz": "CZ",
            "swap": "SWAP",
        }
        
        for op, *args in circuit.operations:
            if op == "measure":
                lines.append(f"circuit.append(cirq.measure(qubits[{args[0]}], key='q{args[0]}'))")
            elif op in gate_map:
                gate_name = gate_map[op]
                if len(args) == 1:
                    lines.append(f"circuit.append(cirq.{gate_name}(qubits[{args[0]}]))")
                elif len(args) == 2:
                    lines.append(f"circuit.append(cirq.{gate_name}(qubits[{args[0]}], qubits[{args[1]}]))")
            elif op in ["rx", "ry", "rz"]:
                axis = op[1].upper()
                lines.append(f"circuit.append(cirq.r{axis.lower()}({args[1]:.4f})(qubits[{args[0]}]))")
        
        return "\n".join(lines)


# Convenience function
def generate_circuit(prompt: str, **kwargs) -> Tuple[Any, str]:
    """
    Convenience function to generate a circuit from a prompt.
    
    Args:
        prompt: Natural language description
        **kwargs: Additional arguments passed to CopilotCircuitGenerator
        
    Returns:
        Tuple of (circuit, description)
    """
    generator = CopilotCircuitGenerator()
    return generator.generate_circuit(prompt, **kwargs)
