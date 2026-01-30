"""
Quantum Gates Library
=====================

Comprehensive collection of quantum gates including:
- Single-qubit gates (Pauli, Hadamard, Phase, Rotation)
- Two-qubit gates (CNOT, CZ, SWAP, controlled rotations)
- Multi-qubit gates (Toffoli, Fredkin)
- Parameterized gates for variational algorithms
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    """Classification of quantum gates."""
    SINGLE = "single"
    TWO_QUBIT = "two_qubit"
    MULTI_QUBIT = "multi_qubit"
    PARAMETERIZED = "parameterized"


@dataclass
class Gate:
    """Represents a quantum gate."""
    name: str
    matrix: np.ndarray
    num_qubits: int
    gate_type: GateType
    params: Optional[dict] = None
    
    def __repr__(self) -> str:
        return f"Gate({self.name}, qubits={self.num_qubits})"


class QuantumGates:
    """
    Library of quantum gates with methods for gate construction
    and multi-qubit expansion.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SINGLE-QUBIT GATES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def I() -> Gate:
        """Identity gate."""
        return Gate(
            name="I",
            matrix=np.eye(2, dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def X() -> Gate:
        """Pauli-X (NOT) gate: |0⟩↔|1⟩"""
        return Gate(
            name="X",
            matrix=np.array([[0, 1], [1, 0]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def Y() -> Gate:
        """Pauli-Y gate."""
        return Gate(
            name="Y",
            matrix=np.array([[0, -1j], [1j, 0]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def Z() -> Gate:
        """Pauli-Z gate: |1⟩ → -|1⟩"""
        return Gate(
            name="Z",
            matrix=np.array([[1, 0], [0, -1]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def H() -> Gate:
        """Hadamard gate: Creates superposition."""
        return Gate(
            name="H",
            matrix=np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def S() -> Gate:
        """S (Phase) gate: √Z"""
        return Gate(
            name="S",
            matrix=np.array([[1, 0], [0, 1j]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def Sdg() -> Gate:
        """S-dagger gate: S†"""
        return Gate(
            name="S†",
            matrix=np.array([[1, 0], [0, -1j]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def T() -> Gate:
        """T gate: √S (π/4 phase)"""
        return Gate(
            name="T",
            matrix=np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    @staticmethod
    def Tdg() -> Gate:
        """T-dagger gate: T†"""
        return Gate(
            name="T†",
            matrix=np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.SINGLE
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ROTATION GATES (Parameterized)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def RX(theta: float) -> Gate:
        """Rotation around X-axis by angle theta."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return Gate(
            name=f"RX({theta:.3f})",
            matrix=np.array([[c, -1j * s], [-1j * s, c]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta, "axis": "X"}
        )
    
    @staticmethod
    def RY(theta: float) -> Gate:
        """Rotation around Y-axis by angle theta."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return Gate(
            name=f"RY({theta:.3f})",
            matrix=np.array([[c, -s], [s, c]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta, "axis": "Y"}
        )
    
    @staticmethod
    def RZ(theta: float) -> Gate:
        """Rotation around Z-axis by angle theta."""
        return Gate(
            name=f"RZ({theta:.3f})",
            matrix=np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ], dtype=complex),
            num_qubits=1,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta, "axis": "Z"}
        )
    
    @staticmethod
    def P(phi: float) -> Gate:
        """Phase gate with arbitrary phase φ."""
        return Gate(
            name=f"P({phi:.3f})",
            matrix=np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex),
            num_qubits=1,
            gate_type=GateType.PARAMETERIZED,
            params={"phi": phi}
        )
    
    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> Gate:
        """
        General single-qubit rotation (U3 gate).
        
        U3(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ)
        """
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return Gate(
            name=f"U3({theta:.2f},{phi:.2f},{lam:.2f})",
            matrix=np.array([
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
            ], dtype=complex),
            num_qubits=1,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta, "phi": phi, "lambda": lam}
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TWO-QUBIT GATES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def CNOT() -> Gate:
        """Controlled-NOT (CX) gate."""
        return Gate(
            name="CNOT",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.TWO_QUBIT
        )
    
    @staticmethod
    def CX() -> Gate:
        """Alias for CNOT."""
        return QuantumGates.CNOT()
    
    @staticmethod
    def CY() -> Gate:
        """Controlled-Y gate."""
        return Gate(
            name="CY",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.TWO_QUBIT
        )
    
    @staticmethod
    def CZ() -> Gate:
        """Controlled-Z gate."""
        return Gate(
            name="CZ",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.TWO_QUBIT
        )
    
    @staticmethod
    def SWAP() -> Gate:
        """SWAP gate: exchanges two qubits."""
        return Gate(
            name="SWAP",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.TWO_QUBIT
        )
    
    @staticmethod
    def iSWAP() -> Gate:
        """iSWAP gate: SWAP with phase."""
        return Gate(
            name="iSWAP",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.TWO_QUBIT
        )
    
    @staticmethod
    def SQSWAP() -> Gate:
        """Square root of SWAP gate."""
        return Gate(
            name="√SWAP",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                [0, 0, 0, 1]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.TWO_QUBIT
        )
    
    @staticmethod
    def CRX(theta: float) -> Gate:
        """Controlled RX rotation."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return Gate(
            name=f"CRX({theta:.3f})",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, c, -1j * s],
                [0, 0, -1j * s, c]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta}
        )
    
    @staticmethod
    def CRY(theta: float) -> Gate:
        """Controlled RY rotation."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return Gate(
            name=f"CRY({theta:.3f})",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, c, -s],
                [0, 0, s, c]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta}
        )
    
    @staticmethod
    def CRZ(theta: float) -> Gate:
        """Controlled RZ rotation."""
        return Gate(
            name=f"CRZ({theta:.3f})",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * theta / 2), 0],
                [0, 0, 0, np.exp(1j * theta / 2)]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta}
        )
    
    @staticmethod
    def CP(phi: float) -> Gate:
        """Controlled Phase gate."""
        return Gate(
            name=f"CP({phi:.3f})",
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * phi)]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.PARAMETERIZED,
            params={"phi": phi}
        )
    
    @staticmethod
    def XX(theta: float) -> Gate:
        """Ising XX coupling gate."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return Gate(
            name=f"XX({theta:.3f})",
            matrix=np.array([
                [c, 0, 0, -1j * s],
                [0, c, -1j * s, 0],
                [0, -1j * s, c, 0],
                [-1j * s, 0, 0, c]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta}
        )
    
    @staticmethod
    def YY(theta: float) -> Gate:
        """Ising YY coupling gate."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return Gate(
            name=f"YY({theta:.3f})",
            matrix=np.array([
                [c, 0, 0, 1j * s],
                [0, c, -1j * s, 0],
                [0, -1j * s, c, 0],
                [1j * s, 0, 0, c]
            ], dtype=complex),
            num_qubits=2,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta}
        )
    
    @staticmethod
    def ZZ(theta: float) -> Gate:
        """Ising ZZ coupling gate."""
        return Gate(
            name=f"ZZ({theta:.3f})",
            matrix=np.diag([
                np.exp(-1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(-1j * theta / 2)
            ]),
            num_qubits=2,
            gate_type=GateType.PARAMETERIZED,
            params={"theta": theta}
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-QUBIT GATES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def Toffoli() -> Gate:
        """Toffoli (CCNOT) gate: 3-qubit controlled-controlled-NOT."""
        matrix = np.eye(8, dtype=complex)
        matrix[6, 6], matrix[6, 7] = 0, 1
        matrix[7, 6], matrix[7, 7] = 1, 0
        return Gate(
            name="Toffoli",
            matrix=matrix,
            num_qubits=3,
            gate_type=GateType.MULTI_QUBIT
        )
    
    @staticmethod
    def CCNOT() -> Gate:
        """Alias for Toffoli gate."""
        return QuantumGates.Toffoli()
    
    @staticmethod
    def Fredkin() -> Gate:
        """Fredkin (CSWAP) gate: controlled SWAP."""
        matrix = np.eye(8, dtype=complex)
        matrix[5, 5], matrix[5, 6] = 0, 1
        matrix[6, 5], matrix[6, 6] = 1, 0
        return Gate(
            name="Fredkin",
            matrix=matrix,
            num_qubits=3,
            gate_type=GateType.MULTI_QUBIT
        )
    
    @staticmethod
    def CSWAP() -> Gate:
        """Alias for Fredkin gate."""
        return QuantumGates.Fredkin()
    
    @staticmethod
    def MCX(num_controls: int) -> Gate:
        """
        Multi-controlled X gate.
        
        Args:
            num_controls: Number of control qubits (1 = CNOT, 2 = Toffoli, etc.)
        """
        n = num_controls + 1
        dim = 2 ** n
        matrix = np.eye(dim, dtype=complex)
        # Flip last two elements
        matrix[dim - 2, dim - 2], matrix[dim - 2, dim - 1] = 0, 1
        matrix[dim - 1, dim - 2], matrix[dim - 1, dim - 1] = 1, 0
        return Gate(
            name=f"MCX({num_controls})",
            matrix=matrix,
            num_qubits=n,
            gate_type=GateType.MULTI_QUBIT
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def expand_gate(gate: Gate, target_qubits: List[int], total_qubits: int) -> np.ndarray:
        """
        Expand a gate to act on specific qubits in a larger system.
        
        Args:
            gate: The gate to expand
            target_qubits: List of qubit indices the gate acts on
            total_qubits: Total number of qubits in the system
            
        Returns:
            Expanded gate matrix operating on the full Hilbert space
        """
        if len(target_qubits) != gate.num_qubits:
            raise ValueError(f"Gate requires {gate.num_qubits} qubits, got {len(target_qubits)}")
        
        if gate.num_qubits == 1:
            return QuantumGates._expand_single_qubit(gate.matrix, target_qubits[0], total_qubits)
        elif gate.num_qubits == 2:
            return QuantumGates._expand_two_qubit(gate.matrix, target_qubits, total_qubits)
        else:
            return QuantumGates._expand_multi_qubit(gate.matrix, target_qubits, total_qubits)
    
    @staticmethod
    def _expand_single_qubit(matrix: np.ndarray, target: int, n: int) -> np.ndarray:
        """Expand single-qubit gate using tensor products."""
        result = np.array([[1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        for i in range(n):
            if i == target:
                result = np.kron(result, matrix)
            else:
                result = np.kron(result, I)
        
        return result
    
    @staticmethod
    def _expand_two_qubit(matrix: np.ndarray, targets: List[int], n: int) -> np.ndarray:
        """Expand two-qubit gate to full Hilbert space."""
        dim = 2 ** n
        expanded = np.zeros((dim, dim), dtype=complex)
        
        control, target = targets[0], targets[1]
        
        for i in range(dim):
            for j in range(dim):
                # Extract control and target bits
                i_ctrl = (i >> (n - 1 - control)) & 1
                i_targ = (i >> (n - 1 - target)) & 1
                j_ctrl = (j >> (n - 1 - control)) & 1
                j_targ = (j >> (n - 1 - target)) & 1
                
                # Check if other bits match
                i_other = i & ~((1 << (n - 1 - control)) | (1 << (n - 1 - target)))
                j_other = j & ~((1 << (n - 1 - control)) | (1 << (n - 1 - target)))
                
                if i_other == j_other:
                    row_idx = i_ctrl * 2 + i_targ
                    col_idx = j_ctrl * 2 + j_targ
                    expanded[i, j] = matrix[row_idx, col_idx]
        
        return expanded
    
    @staticmethod
    def _expand_multi_qubit(matrix: np.ndarray, targets: List[int], n: int) -> np.ndarray:
        """Expand multi-qubit gate to full Hilbert space."""
        dim = 2 ** n
        gate_dim = 2 ** len(targets)
        expanded = np.zeros((dim, dim), dtype=complex)
        
        for i in range(dim):
            for j in range(dim):
                # Extract target qubit values
                i_targets = 0
                j_targets = 0
                for k, t in enumerate(targets):
                    i_targets |= ((i >> (n - 1 - t)) & 1) << (len(targets) - 1 - k)
                    j_targets |= ((j >> (n - 1 - t)) & 1) << (len(targets) - 1 - k)
                
                # Check if non-target qubits match
                mask = 0
                for t in targets:
                    mask |= (1 << (n - 1 - t))
                
                if (i & ~mask) == (j & ~mask):
                    expanded[i, j] = matrix[i_targets, j_targets]
        
        return expanded
    
    @staticmethod
    def create_custom(matrix: np.ndarray, name: str = "Custom") -> Gate:
        """
        Create a custom gate from a unitary matrix.
        
        Validates that the matrix is unitary.
        """
        # Check unitarity: U†U = I
        if not np.allclose(matrix @ np.conj(matrix.T), np.eye(matrix.shape[0])):
            raise ValueError("Matrix is not unitary")
        
        num_qubits = int(np.log2(matrix.shape[0]))
        
        return Gate(
            name=name,
            matrix=matrix,
            num_qubits=num_qubits,
            gate_type=GateType.SINGLE if num_qubits == 1 else GateType.MULTI_QUBIT
        )
    
    @classmethod
    def get_all_single_qubit_gates(cls) -> List[Gate]:
        """Return list of all single-qubit gates."""
        return [
            cls.I(), cls.X(), cls.Y(), cls.Z(),
            cls.H(), cls.S(), cls.Sdg(), cls.T(), cls.Tdg()
        ]
    
    @classmethod
    def get_gate_by_name(cls, name: str, **params) -> Gate:
        """
        Get a gate by its name string.
        
        Args:
            name: Gate name (e.g., "H", "CNOT", "RX")
            **params: Parameters for parameterized gates
        """
        gate_map = {
            "I": cls.I, "X": cls.X, "Y": cls.Y, "Z": cls.Z,
            "H": cls.H, "S": cls.S, "Sdg": cls.Sdg, "T": cls.T, "Tdg": cls.Tdg,
            "CNOT": cls.CNOT, "CX": cls.CX, "CY": cls.CY, "CZ": cls.CZ,
            "SWAP": cls.SWAP, "iSWAP": cls.iSWAP,
            "Toffoli": cls.Toffoli, "CCNOT": cls.CCNOT,
            "Fredkin": cls.Fredkin, "CSWAP": cls.CSWAP,
        }
        
        param_gates = {
            "RX": cls.RX, "RY": cls.RY, "RZ": cls.RZ, "P": cls.P,
            "U3": cls.U3,
            "CRX": cls.CRX, "CRY": cls.CRY, "CRZ": cls.CRZ, "CP": cls.CP,
            "XX": cls.XX, "YY": cls.YY, "ZZ": cls.ZZ,
        }
        
        if name.upper() in gate_map:
            return gate_map[name.upper()]()
        elif name.upper() in param_gates:
            return param_gates[name.upper()](**params)
        else:
            raise ValueError(f"Unknown gate: {name}")
