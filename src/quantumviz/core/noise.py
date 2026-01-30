"""
Quantum Noise Models
====================

Realistic noise modeling for quantum simulations including:
- Depolarizing noise
- Amplitude damping (T1 decay)
- Phase damping (T2 decay)
- Thermal noise
- Custom Kraus operators
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NoiseType(Enum):
    """Types of quantum noise."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    THERMAL = "thermal"
    CUSTOM = "custom"


@dataclass
class NoiseChannel:
    """
    Represents a quantum noise channel using Kraus operator formalism.
    
    A quantum channel ε acts on density matrix ρ as:
    ε(ρ) = Σ_k K_k ρ K_k†
    
    where {K_k} are Kraus operators satisfying Σ_k K_k† K_k = I
    """
    
    name: str
    noise_type: NoiseType
    kraus_operators: List[np.ndarray]
    params: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """Apply the noise channel to a density matrix."""
        result = np.zeros_like(density_matrix)
        
        for K in self.kraus_operators:
            result += K @ density_matrix @ np.conj(K.T)
        
        return result
    
    def validate(self) -> bool:
        """Check that Kraus operators satisfy completeness relation."""
        dim = self.kraus_operators[0].shape[0]
        total = np.zeros((dim, dim), dtype=complex)
        
        for K in self.kraus_operators:
            total += np.conj(K.T) @ K
        
        return np.allclose(total, np.eye(dim))
    
    def __repr__(self) -> str:
        return f"NoiseChannel({self.name}, type={self.noise_type.value})"


class NoiseModel:
    """
    Comprehensive noise model for quantum simulations.
    
    Supports:
    - Per-gate noise
    - Idle/decoherence noise
    - Measurement errors
    - Crosstalk between qubits
    """
    
    def __init__(self):
        """Initialize empty noise model."""
        self.gate_noise: Dict[str, NoiseChannel] = {}
        self.idle_noise: Optional[NoiseChannel] = None
        self.measurement_error: float = 0.0
        self.readout_error: Dict[str, float] = {"0->1": 0.0, "1->0": 0.0}
        self.crosstalk: Dict[Tuple[int, int], float] = {}
        self.t1: Optional[float] = None  # T1 relaxation time
        self.t2: Optional[float] = None  # T2 dephasing time
        self.gate_times: Dict[str, float] = {}  # Gate durations for time evolution
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STANDARD NOISE CHANNELS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def depolarizing_channel(p: float) -> NoiseChannel:
        """
        Depolarizing channel: with probability p, replace state with maximally mixed.
        
        ε(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
        
        Args:
            p: Error probability (0 ≤ p ≤ 1)
        """
        if not 0 <= p <= 1:
            raise ValueError("Probability must be in [0, 1]")
        
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        kraus = [
            np.sqrt(1 - 3*p/4) * I,
            np.sqrt(p/4) * X,
            np.sqrt(p/4) * Y,
            np.sqrt(p/4) * Z
        ]
        
        return NoiseChannel(
            name=f"Depolarizing(p={p:.4f})",
            noise_type=NoiseType.DEPOLARIZING,
            kraus_operators=kraus,
            params={"p": p}
        )
    
    @staticmethod
    def amplitude_damping_channel(gamma: float) -> NoiseChannel:
        """
        Amplitude damping (T1 decay): models energy relaxation.
        
        |1⟩ decays to |0⟩ with probability γ.
        
        Args:
            gamma: Decay probability (0 ≤ γ ≤ 1)
        """
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be in [0, 1]")
        
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        
        return NoiseChannel(
            name=f"AmplitudeDamping(γ={gamma:.4f})",
            noise_type=NoiseType.AMPLITUDE_DAMPING,
            kraus_operators=[K0, K1],
            params={"gamma": gamma}
        )
    
    @staticmethod
    def phase_damping_channel(gamma: float) -> NoiseChannel:
        """
        Phase damping (pure dephasing): models loss of coherence without energy loss.
        
        Args:
            gamma: Dephasing probability (0 ≤ γ ≤ 1)
        """
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be in [0, 1]")
        
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
        
        return NoiseChannel(
            name=f"PhaseDamping(γ={gamma:.4f})",
            noise_type=NoiseType.PHASE_DAMPING,
            kraus_operators=[K0, K1],
            params={"gamma": gamma}
        )
    
    @staticmethod
    def bit_flip_channel(p: float) -> NoiseChannel:
        """
        Bit flip channel: |0⟩↔|1⟩ with probability p.
        
        Args:
            p: Flip probability
        """
        if not 0 <= p <= 1:
            raise ValueError("Probability must be in [0, 1]")
        
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        
        return NoiseChannel(
            name=f"BitFlip(p={p:.4f})",
            noise_type=NoiseType.BIT_FLIP,
            kraus_operators=[np.sqrt(1 - p) * I, np.sqrt(p) * X],
            params={"p": p}
        )
    
    @staticmethod
    def phase_flip_channel(p: float) -> NoiseChannel:
        """
        Phase flip channel: Z with probability p.
        
        Args:
            p: Flip probability
        """
        if not 0 <= p <= 1:
            raise ValueError("Probability must be in [0, 1]")
        
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        return NoiseChannel(
            name=f"PhaseFlip(p={p:.4f})",
            noise_type=NoiseType.PHASE_FLIP,
            kraus_operators=[np.sqrt(1 - p) * I, np.sqrt(p) * Z],
            params={"p": p}
        )
    
    @staticmethod
    def thermal_relaxation_channel(t1: float, t2: float, time: float, excited_pop: float = 0.0) -> NoiseChannel:
        """
        Combined T1/T2 thermal relaxation channel.
        
        Models realistic qubit decoherence with:
        - Energy relaxation (T1)
        - Pure dephasing (T2)
        - Thermal excitation (for non-zero temperature)
        
        Args:
            t1: T1 relaxation time
            t2: T2 dephasing time (must be ≤ 2*T1)
            time: Evolution time
            excited_pop: Thermal excited state population (temperature dependent)
        """
        if t2 > 2 * t1:
            raise ValueError("T2 must be ≤ 2*T1")
        
        # Decay probabilities
        p_reset = 1 - np.exp(-time / t1)
        p_z = (1 - p_reset) * (1 - np.exp(-time / t2 + time / (2 * t1))) / 2
        
        # Population after relaxation
        p0 = (1 - excited_pop) * p_reset + (1 - p_reset)  # prob to be in |0⟩
        p1 = excited_pop * p_reset + (1 - p_reset) * p_z
        
        # Construct Kraus operators
        kraus = []
        
        # Identity contribution
        if (1 - p_reset - 2*p_z) > 0:
            kraus.append(np.sqrt(1 - p_reset - 2*p_z) * np.eye(2, dtype=complex))
        
        # Reset to |0⟩
        kraus.append(np.sqrt(p_reset * (1 - excited_pop)) * np.array([[1, 0], [0, 0]], dtype=complex))
        kraus.append(np.sqrt(p_reset * (1 - excited_pop)) * np.array([[0, 1], [0, 0]], dtype=complex))
        
        # Reset to |1⟩ (thermal excitation)
        if excited_pop > 0:
            kraus.append(np.sqrt(p_reset * excited_pop) * np.array([[0, 0], [1, 0]], dtype=complex))
            kraus.append(np.sqrt(p_reset * excited_pop) * np.array([[0, 0], [0, 1]], dtype=complex))
        
        # Z error (dephasing)
        if p_z > 0:
            kraus.append(np.sqrt(p_z) * np.array([[1, 0], [0, -1]], dtype=complex))
        
        # Filter out zero operators
        kraus = [K for K in kraus if np.linalg.norm(K) > 1e-10]
        
        return NoiseChannel(
            name=f"ThermalRelaxation(T1={t1:.2f}, T2={t2:.2f}, t={time:.4f})",
            noise_type=NoiseType.THERMAL,
            kraus_operators=kraus,
            params={"t1": t1, "t2": t2, "time": time, "excited_pop": excited_pop}
        )
    
    @staticmethod
    def custom_channel(kraus_operators: List[np.ndarray], name: str = "Custom") -> NoiseChannel:
        """
        Create a custom noise channel from Kraus operators.
        
        Args:
            kraus_operators: List of Kraus operator matrices
            name: Channel name
        """
        channel = NoiseChannel(
            name=name,
            noise_type=NoiseType.CUSTOM,
            kraus_operators=kraus_operators
        )
        
        if not channel.validate():
            raise ValueError("Kraus operators don't satisfy completeness relation")
        
        return channel
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-QUBIT NOISE
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def two_qubit_depolarizing(p: float) -> NoiseChannel:
        """
        Two-qubit depolarizing channel.
        
        Args:
            p: Error probability
        """
        # Pauli basis for 2 qubits
        I = np.eye(2, dtype=complex)
        paulis_1 = [I, 
                    np.array([[0, 1], [1, 0]], dtype=complex),
                    np.array([[0, -1j], [1j, 0]], dtype=complex),
                    np.array([[1, 0], [0, -1]], dtype=complex)]
        
        paulis_2 = [np.kron(p1, p2) for p1 in paulis_1 for p2 in paulis_1]
        
        kraus = [np.sqrt(1 - 15*p/16) * paulis_2[0]]  # Identity
        kraus.extend([np.sqrt(p/16) * P for P in paulis_2[1:]])  # Error terms
        
        return NoiseChannel(
            name=f"TwoQubitDepolarizing(p={p:.4f})",
            noise_type=NoiseType.DEPOLARIZING,
            kraus_operators=kraus,
            params={"p": p}
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NOISE MODEL CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_gate_noise(self, gate_name: str, channel: NoiseChannel):
        """Add noise to a specific gate type."""
        self.gate_noise[gate_name] = channel
    
    def set_idle_noise(self, channel: NoiseChannel):
        """Set noise for idle qubits between gates."""
        self.idle_noise = channel
    
    def set_t1_t2(self, t1: float, t2: float):
        """Set T1 and T2 times for automatic thermal noise calculation."""
        if t2 > 2 * t1:
            raise ValueError("T2 must be ≤ 2*T1")
        self.t1 = t1
        self.t2 = t2
    
    def set_gate_time(self, gate_name: str, time: float):
        """Set execution time for a gate (for T1/T2 calculations)."""
        self.gate_times[gate_name] = time
    
    def set_measurement_error(self, error: float):
        """Set symmetric measurement error rate."""
        self.measurement_error = error
        self.readout_error = {"0->1": error, "1->0": error}
    
    def set_readout_errors(self, p_01: float, p_10: float):
        """Set asymmetric readout errors."""
        self.readout_error["0->1"] = p_01
        self.readout_error["1->0"] = p_10
    
    def set_crosstalk(self, qubit1: int, qubit2: int, strength: float):
        """Set crosstalk between two qubits."""
        self.crosstalk[(qubit1, qubit2)] = strength
        self.crosstalk[(qubit2, qubit1)] = strength
    
    def get_noise_for_gate(self, gate_name: str, gate_time: Optional[float] = None) -> Optional[NoiseChannel]:
        """
        Get the appropriate noise channel for a gate operation.
        
        If T1/T2 times are set and gate time is known, automatically generates
        thermal relaxation noise.
        """
        # Check for explicit gate noise
        if gate_name in self.gate_noise:
            return self.gate_noise[gate_name]
        
        # Generate thermal noise if T1/T2 are set
        if self.t1 is not None and self.t2 is not None:
            time = gate_time or self.gate_times.get(gate_name, 0.0)
            if time > 0:
                return self.thermal_relaxation_channel(self.t1, self.t2, time)
        
        return None
    
    def apply_measurement_error(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Apply readout errors to measurement counts."""
        if self.measurement_error == 0 and all(v == 0 for v in self.readout_error.values()):
            return counts
        
        noisy_counts = {}
        
        for bitstring, count in counts.items():
            for _ in range(count):
                noisy_bits = list(bitstring)
                for i, bit in enumerate(noisy_bits):
                    if np.random.random() < self.readout_error[f"{bit}->{'1' if bit == '0' else '0'}"]:
                        noisy_bits[i] = '1' if bit == '0' else '0'
                
                noisy_string = ''.join(noisy_bits)
                noisy_counts[noisy_string] = noisy_counts.get(noisy_string, 0) + 1
        
        return noisy_counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize noise model to dictionary."""
        return {
            "t1": self.t1,
            "t2": self.t2,
            "measurement_error": self.measurement_error,
            "readout_error": self.readout_error,
            "gate_times": self.gate_times,
            "gate_noise": {k: v.name for k, v in self.gate_noise.items()},
        }
    
    @classmethod
    def from_backend_properties(cls, t1: float, t2: float, 
                                 single_qubit_error: float = 0.001,
                                 two_qubit_error: float = 0.01,
                                 readout_error: float = 0.02) -> "NoiseModel":
        """
        Create a realistic noise model from typical backend parameters.
        
        Args:
            t1: T1 relaxation time (microseconds)
            t2: T2 dephasing time (microseconds)
            single_qubit_error: Single-qubit gate error rate
            two_qubit_error: Two-qubit gate error rate
            readout_error: Measurement error rate
        """
        model = cls()
        model.set_t1_t2(t1, t2)
        
        # Typical gate times (in microseconds)
        model.set_gate_time("single", 0.05)  # 50 ns
        model.set_gate_time("two_qubit", 0.3)  # 300 ns
        model.set_gate_time("measurement", 1.0)  # 1 μs
        
        # Add depolarizing noise for gates
        model.add_gate_noise("X", cls.depolarizing_channel(single_qubit_error))
        model.add_gate_noise("Y", cls.depolarizing_channel(single_qubit_error))
        model.add_gate_noise("Z", cls.depolarizing_channel(single_qubit_error))
        model.add_gate_noise("H", cls.depolarizing_channel(single_qubit_error))
        model.add_gate_noise("CNOT", cls.two_qubit_depolarizing(two_qubit_error))
        model.add_gate_noise("CZ", cls.two_qubit_depolarizing(two_qubit_error))
        
        model.set_measurement_error(readout_error)
        
        return model
    
    @classmethod
    def ideal(cls) -> "NoiseModel":
        """Create an ideal (noiseless) model."""
        return cls()
    
    @classmethod
    def noisy_simulator(cls, noise_level: str = "moderate") -> "NoiseModel":
        """
        Create a pre-configured noise model.
        
        Args:
            noise_level: One of "low", "moderate", "high"
        """
        levels = {
            "low": (100.0, 80.0, 0.0005, 0.005, 0.01),
            "moderate": (50.0, 40.0, 0.001, 0.01, 0.02),
            "high": (20.0, 15.0, 0.005, 0.05, 0.05),
        }
        
        if noise_level not in levels:
            raise ValueError(f"Unknown noise level: {noise_level}")
        
        t1, t2, sq_err, tq_err, readout = levels[noise_level]
        return cls.from_backend_properties(t1, t2, sq_err, tq_err, readout)
    
    def __repr__(self) -> str:
        parts = [f"NoiseModel(T1={self.t1}, T2={self.t2}"]
        if self.gate_noise:
            parts.append(f"gates={list(self.gate_noise.keys())}")
        if self.measurement_error > 0:
            parts.append(f"meas_err={self.measurement_error:.4f}")
        return ", ".join(parts) + ")"
