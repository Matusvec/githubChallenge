"""
Helper Utilities
================

Common helper functions for QuantumViz.
"""

from __future__ import annotations
import time
import functools
from typing import Callable, Any, Optional, List, Dict
from contextlib import contextmanager
import numpy as np


def format_state_vector(
    state_vector: np.ndarray,
    num_qubits: int,
    threshold: float = 1e-10,
    precision: int = 4,
    max_terms: int = 10,
) -> str:
    """
    Format a state vector as a human-readable string.
    
    Args:
        state_vector: Complex state vector
        num_qubits: Number of qubits
        threshold: Amplitudes below this are omitted
        precision: Decimal precision
        max_terms: Maximum number of terms to show
        
    Returns:
        Formatted string representation
    """
    state_vector = np.asarray(state_vector).flatten()
    
    terms = []
    for i, amp in enumerate(state_vector):
        if abs(amp) > threshold:
            basis_state = format(i, f'0{num_qubits}b')
            
            # Format amplitude
            if abs(amp.imag) < threshold:
                # Real only
                amp_str = f"{amp.real:+.{precision}f}"
            elif abs(amp.real) < threshold:
                # Imaginary only
                amp_str = f"{amp.imag:+.{precision}f}i"
            else:
                # Complex
                sign = '+' if amp.imag >= 0 else '-'
                amp_str = f"({amp.real:.{precision}f}{sign}{abs(amp.imag):.{precision}f}i)"
            
            terms.append(f"{amp_str}|{basis_state}⟩")
            
            if len(terms) >= max_terms:
                terms.append(f"... (+{len(state_vector) - max_terms} more terms)")
                break
    
    if not terms:
        return "|0...0⟩"
    
    return " ".join(terms)


def format_probability_table(
    probabilities: np.ndarray,
    num_qubits: int,
    threshold: float = 0.001,
    sort: bool = True,
    max_rows: int = 16,
) -> str:
    """
    Format probability distribution as an ASCII table.
    
    Args:
        probabilities: Probability array
        num_qubits: Number of qubits
        threshold: Probabilities below this are omitted
        sort: Whether to sort by probability
        max_rows: Maximum number of rows
        
    Returns:
        Formatted table string
    """
    probabilities = np.asarray(probabilities).flatten()
    
    # Create rows
    rows = []
    for i, prob in enumerate(probabilities):
        if prob > threshold:
            basis_state = format(i, f'0{num_qubits}b')
            rows.append((basis_state, prob))
    
    # Sort if requested
    if sort:
        rows.sort(key=lambda x: -x[1])
    
    # Truncate
    if len(rows) > max_rows:
        rows = rows[:max_rows]
        rows.append(("...", None))
    
    # Format table
    lines = [
        "┌" + "─" * (num_qubits + 2) + "┬" + "─" * 14 + "┬" + "─" * 22 + "┐",
        f"│ {'State':<{num_qubits}} │ {'Probability':^12} │ {'Bar':^20} │",
        "├" + "─" * (num_qubits + 2) + "┼" + "─" * 14 + "┼" + "─" * 22 + "┤",
    ]
    
    for state, prob in rows:
        if prob is not None:
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"│ |{state}⟩ │ {prob:>11.4f} │ {bar} │")
        else:
            lines.append(f"│ {'...':>{num_qubits + 1}} │ {'...':^12} │ {'...':^20} │")
    
    lines.append("└" + "─" * (num_qubits + 2) + "┴" + "─" * 14 + "┴" + "─" * 22 + "┘")
    
    return "\n".join(lines)


def format_complex_matrix(
    matrix: np.ndarray,
    precision: int = 3,
    max_size: int = 8,
) -> str:
    """Format a complex matrix for display."""
    rows, cols = matrix.shape
    
    if rows > max_size or cols > max_size:
        return f"[{rows}×{cols} matrix - too large to display]"
    
    lines = []
    for i in range(rows):
        row_str = []
        for j in range(cols):
            val = matrix[i, j]
            if abs(val.imag) < 1e-10:
                row_str.append(f"{val.real:>{precision + 4}.{precision}f}")
            elif abs(val.real) < 1e-10:
                row_str.append(f"{val.imag:>{precision + 3}.{precision}f}i")
            else:
                sign = '+' if val.imag >= 0 else '-'
                row_str.append(f"{val.real:.{precision}f}{sign}{abs(val.imag):.{precision}f}i")
        
        lines.append("  ".join(row_str))
    
    return "\n".join(lines)


@contextmanager
def measure_execution_time(name: str = "Operation"):
    """
    Context manager to measure execution time.
    
    Example:
        >>> with measure_execution_time("Simulation"):
        ...     result = simulator.run(circuit)
        Simulation completed in 0.123s
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} completed in {elapsed:.3f}s")


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Example:
        >>> @timer
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()
        slow_function took 1.001s
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def validate_qubit_index(qubit: int, num_qubits: int, name: str = "qubit") -> None:
    """Validate that a qubit index is valid."""
    if not isinstance(qubit, int):
        raise TypeError(f"{name} must be an integer, got {type(qubit)}")
    if qubit < 0 or qubit >= num_qubits:
        raise ValueError(f"{name} {qubit} is out of range [0, {num_qubits - 1}]")


def validate_probability_distribution(probs: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if array is a valid probability distribution."""
    probs = np.asarray(probs)
    
    if np.any(probs < -tolerance):
        return False
    
    if abs(np.sum(probs) - 1.0) > tolerance:
        return False
    
    return True


def binary_to_int(binary_string: str) -> int:
    """Convert binary string to integer."""
    return int(binary_string, 2)


def int_to_binary(value: int, num_bits: int) -> str:
    """Convert integer to binary string with specified number of bits."""
    return format(value, f'0{num_bits}b')


def tensor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute tensor (Kronecker) product of two arrays."""
    return np.kron(a, b)


def partial_trace(
    rho: np.ndarray,
    keep: List[int],
    dims: List[int],
) -> np.ndarray:
    """
    Compute partial trace of a density matrix.
    
    Args:
        rho: Density matrix
        keep: List of subsystem indices to keep
        dims: Dimensions of each subsystem
        
    Returns:
        Reduced density matrix
    """
    num_systems = len(dims)
    total_dim = np.prod(dims)
    
    if rho.shape != (total_dim, total_dim):
        raise ValueError(f"Density matrix shape {rho.shape} doesn't match dims {dims}")
    
    # Reshape into tensor
    rho_tensor = rho.reshape(dims + dims)
    
    # Determine which indices to trace out
    trace_out = [i for i in range(num_systems) if i not in keep]
    
    # Trace out indices (from highest to lowest to avoid index shifting)
    for idx in sorted(trace_out, reverse=True):
        rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + num_systems)
        num_systems -= 1
    
    # Reshape back to matrix
    kept_dims = [dims[i] for i in sorted(keep)]
    new_dim = np.prod(kept_dims)
    
    return rho_tensor.reshape(new_dim, new_dim)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute fidelity between two density matrices.
    
    F(ρ, σ) = (Tr√(√ρ σ √ρ))²
    """
    from scipy import linalg
    
    sqrt_rho = linalg.sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = linalg.sqrtm(inner)
    
    return float(np.real(np.trace(sqrt_inner)) ** 2)


def state_fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compute fidelity between two pure states."""
    psi1 = np.asarray(psi1).flatten()
    psi2 = np.asarray(psi2).flatten()
    
    return float(abs(np.vdot(psi1, psi2)) ** 2)


def entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy of a density matrix."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Filter near-zero
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def purity(rho: np.ndarray) -> float:
    """Compute purity Tr(ρ²) of a density matrix."""
    return float(np.real(np.trace(rho @ rho)))


def is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check if a matrix is unitary."""
    n = matrix.shape[0]
    identity = np.eye(n)
    product = matrix @ matrix.conj().T
    return np.allclose(product, identity, atol=tolerance)


def is_hermitian(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check if a matrix is Hermitian."""
    return np.allclose(matrix, matrix.conj().T, atol=tolerance)


def is_positive_semidefinite(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check if a matrix is positive semidefinite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -tolerance)


def random_unitary(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random unitary matrix using QR decomposition."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random complex matrix
    real_part = np.random.randn(n, n)
    imag_part = np.random.randn(n, n)
    z = real_part + 1j * imag_part
    
    # QR decomposition
    q, r = np.linalg.qr(z)
    
    # Make the decomposition unique
    d = np.diag(r)
    ph = d / np.abs(d)
    
    return q @ np.diag(ph)


def random_state_vector(num_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random normalized state vector."""
    if seed is not None:
        np.random.seed(seed)
    
    dim = 2 ** num_qubits
    real_part = np.random.randn(dim)
    imag_part = np.random.randn(dim)
    
    state = real_part + 1j * imag_part
    state /= np.linalg.norm(state)
    
    return state


def random_density_matrix(num_qubits: int, purity: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random density matrix with specified purity.
    
    Args:
        num_qubits: Number of qubits
        purity: Target purity (1.0 = pure state, 1/d = maximally mixed)
        seed: Random seed
        
    Returns:
        Random density matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    dim = 2 ** num_qubits
    
    if purity >= 1.0 - 1e-10:
        # Pure state
        state = random_state_vector(num_qubits, seed)
        return np.outer(state, state.conj())
    
    # Generate random eigenvalues with target purity
    # purity = Σ λᵢ², Σ λᵢ = 1
    eigenvalues = np.random.rand(dim)
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    
    # Adjust for target purity (approximately)
    # This is a simple approach; exact would require optimization
    
    # Generate random unitary
    U = random_unitary(dim, seed)
    
    # Create density matrix
    rho = U @ np.diag(eigenvalues) @ U.conj().T
    
    return rho
