"""
Quantum State Management
========================

Core quantum state representation using QuTiP's density matrix formalism.
Supports pure states, mixed states, and multi-qubit entangled systems.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False


class StateType(Enum):
    """Types of quantum states."""
    PURE = "pure"
    MIXED = "mixed"
    ENTANGLED = "entangled"


@dataclass
class BlochCoordinates:
    """Bloch sphere coordinates for single-qubit visualization."""
    x: float
    y: float
    z: float
    theta: float = 0.0  # Polar angle
    phi: float = 0.0    # Azimuthal angle
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "theta": self.theta, "phi": self.phi}
    
    @classmethod
    def from_state_vector(cls, alpha: complex, beta: complex) -> "BlochCoordinates":
        """Convert |ψ⟩ = α|0⟩ + β|1⟩ to Bloch coordinates."""
        # Normalize
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        if norm < 1e-10:
            return cls(0, 0, 1)  # Default to |0⟩
        
        alpha, beta = alpha / norm, beta / norm
        
        # Bloch vector components
        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = np.abs(alpha)**2 - np.abs(beta)**2
        
        # Spherical coordinates
        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)
        
        return cls(x=float(x), y=float(y), z=float(z), theta=float(theta), phi=float(phi))


@dataclass
class QuantumState:
    """
    Represents a quantum state supporting multiple qubits.
    
    Features:
    - Pure and mixed state support via density matrices
    - Entanglement detection and quantification
    - Bloch sphere coordinates extraction
    - State evolution tracking
    - JSON serialization for export
    """
    
    num_qubits: int
    state_vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    name: str = "quantum_state"
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize state after dataclass creation."""
        if self.state_vector is None and self.density_matrix is None:
            # Default to |0...0⟩ state
            dim = 2 ** self.num_qubits
            self.state_vector = np.zeros(dim, dtype=complex)
            self.state_vector[0] = 1.0
        
        if self.density_matrix is None and self.state_vector is not None:
            # Compute density matrix from state vector
            self.density_matrix = np.outer(self.state_vector, np.conj(self.state_vector))
        
        self._record_state("initialization")
    
    def _record_state(self, operation: str):
        """Record state in history for animation."""
        self.history.append({
            "operation": operation,
            "density_matrix": self.density_matrix.copy() if self.density_matrix is not None else None,
            "state_vector": self.state_vector.copy() if self.state_vector is not None else None,
        })
    
    @property
    def dimension(self) -> int:
        """Hilbert space dimension."""
        return 2 ** self.num_qubits
    
    @property
    def state_type(self) -> StateType:
        """Determine the type of quantum state."""
        if self.density_matrix is None:
            return StateType.PURE
        
        # Check purity: Tr(ρ²) = 1 for pure states
        purity = np.real(np.trace(self.density_matrix @ self.density_matrix))
        
        if np.isclose(purity, 1.0, atol=1e-6):
            if self.num_qubits > 1 and self._is_entangled():
                return StateType.ENTANGLED
            return StateType.PURE
        return StateType.MIXED
    
    def _is_entangled(self) -> bool:
        """Check if multi-qubit state is entangled using partial trace."""
        if self.num_qubits < 2 or self.density_matrix is None:
            return False
        
        try:
            # Compute reduced density matrix by tracing out subsystems
            rho_reduced = self._partial_trace(list(range(1, self.num_qubits)))
            purity_reduced = np.real(np.trace(rho_reduced @ rho_reduced))
            
            # If reduced state is mixed (purity < 1), the full state is entangled
            return purity_reduced < 0.99
        except Exception:
            return False
    
    def _partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """
        Compute partial trace over specified qubits.
        
        Args:
            keep_qubits: Indices of qubits to keep (trace out the rest)
            
        Returns:
            Reduced density matrix
        """
        if self.density_matrix is None:
            raise ValueError("No density matrix available")
        
        n = self.num_qubits
        dims = [2] * n
        
        # Reshape to tensor form
        rho_tensor = self.density_matrix.reshape(dims + dims)
        
        # Determine which indices to trace out
        trace_qubits = [i for i in range(n) if i not in keep_qubits]
        
        # Perform partial trace
        for i, qubit in enumerate(sorted(trace_qubits, reverse=True)):
            # Trace over qubit by summing diagonal elements
            axis1 = qubit
            axis2 = qubit + n - i
            rho_tensor = np.trace(rho_tensor, axis1=axis1, axis2=axis2)
        
        # Reshape back to matrix
        keep_dim = 2 ** len(keep_qubits)
        return rho_tensor.reshape(keep_dim, keep_dim)
    
    def get_bloch_coordinates(self, qubit_index: int = 0) -> BlochCoordinates:
        """
        Extract Bloch sphere coordinates for a specific qubit.
        
        For multi-qubit states, performs partial trace to get single-qubit
        reduced density matrix, then extracts Bloch vector.
        """
        if self.num_qubits == 1:
            rho = self.density_matrix
        else:
            # Get reduced density matrix for the specified qubit
            keep_qubits = [qubit_index]
            rho = self._partial_trace(keep_qubits)
        
        # Extract Bloch vector from 2x2 density matrix
        # ρ = (I + r⃗·σ⃗)/2 where σ⃗ are Pauli matrices
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[1, 0])  # Note: rho[1,0] for proper sign
        z = np.real(rho[0, 0] - rho[1, 1])
        
        # Spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            theta, phi = 0, 0
        else:
            theta = np.arccos(np.clip(z / r, -1, 1))
            phi = np.arctan2(y, x)
        
        return BlochCoordinates(
            x=float(x), y=float(y), z=float(z),
            theta=float(theta), phi=float(phi)
        )
    
    def get_all_bloch_coordinates(self) -> List[BlochCoordinates]:
        """Get Bloch coordinates for all qubits."""
        return [self.get_bloch_coordinates(i) for i in range(self.num_qubits)]
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities in computational basis."""
        if self.density_matrix is not None:
            return np.real(np.diag(self.density_matrix))
        elif self.state_vector is not None:
            return np.abs(self.state_vector) ** 2
        return np.zeros(self.dimension)
    
    def measure(self, shots: int = 1000) -> Dict[str, int]:
        """
        Simulate quantum measurement.
        
        Returns:
            Dictionary mapping basis states (e.g., "010") to counts
        """
        probs = self.get_probabilities()
        outcomes = np.random.choice(self.dimension, size=shots, p=probs)
        
        counts = {}
        for outcome in outcomes:
            basis = format(outcome, f'0{self.num_qubits}b')
            counts[basis] = counts.get(basis, 0) + 1
        
        return dict(sorted(counts.items()))
    
    def apply_operator(self, operator: np.ndarray, record: bool = True, operation_name: str = "gate"):
        """
        Apply a quantum operator to the state.
        
        Args:
            operator: Unitary operator matrix
            record: Whether to record in history
            operation_name: Name for history tracking
        """
        if self.density_matrix is not None:
            # ρ' = U ρ U†
            self.density_matrix = operator @ self.density_matrix @ np.conj(operator.T)
        
        if self.state_vector is not None:
            self.state_vector = operator @ self.state_vector
        
        if record:
            self._record_state(operation_name)
    
    def fidelity(self, other: "QuantumState") -> float:
        """
        Compute quantum fidelity between this state and another.
        
        F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²
        """
        if self.density_matrix is None or other.density_matrix is None:
            raise ValueError("Both states need density matrices")
        
        from scipy.linalg import sqrtm
        
        sqrt_rho = sqrtm(self.density_matrix)
        product = sqrt_rho @ other.density_matrix @ sqrt_rho
        sqrt_product = sqrtm(product)
        
        return float(np.real(np.trace(sqrt_product)) ** 2)
    
    def entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).
        
        Returns 0 for pure states, > 0 for mixed states.
        """
        if self.density_matrix is None:
            return 0.0
        
        eigenvalues = np.linalg.eigvalsh(self.density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove zeros
        
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def concurrence(self) -> float:
        """
        Compute concurrence for 2-qubit states (entanglement measure).
        
        Returns value between 0 (separable) and 1 (maximally entangled).
        """
        if self.num_qubits != 2:
            raise ValueError("Concurrence only defined for 2-qubit states")
        
        if self.density_matrix is None:
            return 0.0
        
        # Pauli Y tensor product
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_yy = np.kron(sigma_y, sigma_y)
        
        # Spin-flipped density matrix
        rho_tilde = sigma_yy @ np.conj(self.density_matrix) @ sigma_yy
        
        # Compute R = sqrt(sqrt(rho) * rho_tilde * sqrt(rho))
        from scipy.linalg import sqrtm
        sqrt_rho = sqrtm(self.density_matrix)
        R = sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)
        
        # Eigenvalues in decreasing order
        eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
        
        # Concurrence
        C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
        return float(C)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "state_type": self.state_type.value,
            "probabilities": self.get_probabilities().tolist(),
            "bloch_coordinates": [bc.to_dict() for bc in self.get_all_bloch_coordinates()],
            "entropy": self.entropy(),
            "purity": float(np.real(np.trace(self.density_matrix @ self.density_matrix))) if self.density_matrix is not None else 1.0,
        }
    
    def to_json(self) -> str:
        """Export state as JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_basis(cls, basis_state: str, name: str = "basis_state") -> "QuantumState":
        """
        Create state from computational basis string.
        
        Example: QuantumState.from_basis("010") creates |010⟩
        """
        n = len(basis_state)
        index = int(basis_state, 2)
        dim = 2 ** n
        
        sv = np.zeros(dim, dtype=complex)
        sv[index] = 1.0
        
        return cls(num_qubits=n, state_vector=sv, name=name)
    
    @classmethod
    def bell_state(cls, which: str = "phi+") -> "QuantumState":
        """
        Create one of the four Bell states.
        
        Args:
            which: One of "phi+", "phi-", "psi+", "psi-"
        """
        sv = np.zeros(4, dtype=complex)
        
        if which == "phi+":  # (|00⟩ + |11⟩) / √2
            sv[0] = sv[3] = 1 / np.sqrt(2)
        elif which == "phi-":  # (|00⟩ - |11⟩) / √2
            sv[0] = 1 / np.sqrt(2)
            sv[3] = -1 / np.sqrt(2)
        elif which == "psi+":  # (|01⟩ + |10⟩) / √2
            sv[1] = sv[2] = 1 / np.sqrt(2)
        elif which == "psi-":  # (|01⟩ - |10⟩) / √2
            sv[1] = 1 / np.sqrt(2)
            sv[2] = -1 / np.sqrt(2)
        else:
            raise ValueError(f"Unknown Bell state: {which}")
        
        return cls(num_qubits=2, state_vector=sv, name=f"bell_{which}")
    
    @classmethod
    def ghz_state(cls, n: int = 3) -> "QuantumState":
        """
        Create n-qubit GHZ state: (|00...0⟩ + |11...1⟩) / √2
        """
        dim = 2 ** n
        sv = np.zeros(dim, dtype=complex)
        sv[0] = sv[-1] = 1 / np.sqrt(2)
        
        return cls(num_qubits=n, state_vector=sv, name=f"ghz_{n}")
    
    @classmethod
    def w_state(cls, n: int = 3) -> "QuantumState":
        """
        Create n-qubit W state: (|100..⟩ + |010..⟩ + ... + |...01⟩) / √n
        """
        dim = 2 ** n
        sv = np.zeros(dim, dtype=complex)
        
        for i in range(n):
            index = 1 << (n - 1 - i)  # Single 1 in position i
            sv[index] = 1 / np.sqrt(n)
        
        return cls(num_qubits=n, state_vector=sv, name=f"w_{n}")
    
    @classmethod
    def random_state(cls, n: int = 1, pure: bool = True, name: str = "random") -> "QuantumState":
        """
        Generate a random quantum state.
        
        Args:
            n: Number of qubits
            pure: If True, generate pure state; else mixed state
            name: State identifier
        """
        dim = 2 ** n
        
        if pure:
            # Random pure state via Haar measure
            real = np.random.randn(dim)
            imag = np.random.randn(dim)
            sv = (real + 1j * imag)
            sv /= np.linalg.norm(sv)
            return cls(num_qubits=n, state_vector=sv, name=name)
        else:
            # Random mixed state via partial trace of larger pure state
            dim_env = 4  # Environment dimension
            total_dim = dim * dim_env
            
            real = np.random.randn(total_dim)
            imag = np.random.randn(total_dim)
            psi = (real + 1j * imag)
            psi /= np.linalg.norm(psi)
            
            # Full density matrix
            rho_full = np.outer(psi, np.conj(psi)).reshape(dim, dim_env, dim, dim_env)
            
            # Partial trace over environment
            rho = np.trace(rho_full, axis1=1, axis2=3)
            
            return cls(num_qubits=n, density_matrix=rho, name=name)
    
    def __repr__(self) -> str:
        return f"QuantumState(qubits={self.num_qubits}, type={self.state_type.value}, name='{self.name}')"
    
    def __str__(self) -> str:
        probs = self.get_probabilities()
        lines = [f"QuantumState '{self.name}' ({self.num_qubits} qubits, {self.state_type.value})"]
        lines.append("-" * 50)
        
        # Show non-zero amplitudes
        for i, p in enumerate(probs):
            if p > 0.001:
                basis = format(i, f'0{self.num_qubits}b')
                lines.append(f"  |{basis}⟩: {p:.4f} ({p*100:.1f}%)")
        
        lines.append("-" * 50)
        lines.append(f"Entropy: {self.entropy():.4f}")
        
        if self.num_qubits == 2:
            lines.append(f"Concurrence: {self.concurrence():.4f}")
        
        return "\n".join(lines)
