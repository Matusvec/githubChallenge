"""
QuantumViz Examples: Basic Usage
================================

This example demonstrates the basic functionality of QuantumViz CLI,
including creating quantum states, running simulations, and visualization.
"""

from quantumviz import (
    QuantumState,
    QuantumCircuit,
    QuantumSimulator,
    BlochSphereRenderer,
    VisualizationTheme,
)


def example_bell_state():
    """Create and visualize a Bell state."""
    print("=" * 60)
    print("Example 1: Bell State")
    print("=" * 60)
    
    # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    bell = QuantumState.bell_state('phi+')
    
    print(f"\nBell State |Φ+⟩:")
    print(bell)
    
    # Check entanglement
    print(f"\nConcurrence: {bell.concurrence():.4f} (1.0 = maximally entangled)")
    print(f"Entropy: {bell.entropy():.4f}")
    
    # Visualize on Bloch spheres
    theme = VisualizationTheme.get_theme("quantum_dark")
    renderer = BlochSphereRenderer(theme=theme)
    
    fig = renderer.render_multi_qubit(bell, title="Bell State |Φ+⟩")
    renderer.save(fig, "bell_state.png")
    print("\nSaved visualization to 'bell_state.png'")
    
    return bell


def example_ghz_state():
    """Create and visualize a GHZ state."""
    print("\n" + "=" * 60)
    print("Example 2: GHZ State (3 qubits)")
    print("=" * 60)
    
    # Create GHZ state (|000⟩ + |111⟩)/√2
    ghz = QuantumState.ghz_state(3)
    
    print(f"\nGHZ State (3 qubits):")
    print(ghz)
    
    # Show probabilities
    probs = ghz.get_probabilities()
    print("\nProbabilities:")
    for i, p in enumerate(probs):
        if p > 0.001:
            print(f"  |{format(i, '03b')}⟩: {p:.4f}")
    
    return ghz


def example_custom_circuit():
    """Build and simulate a custom quantum circuit."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Circuit")
    print("=" * 60)
    
    # Build a circuit
    circuit = QuantumCircuit(3, name="Custom Circuit")
    
    # Create superposition
    circuit.h(0)
    circuit.h(1)
    
    # Entangle qubits
    circuit.cnot(0, 2)
    circuit.cnot(1, 2)
    
    # Add some rotations
    import numpy as np
    circuit.rz(2, np.pi / 4)
    
    # Measure
    circuit.measure_all()
    
    print(f"\nCircuit: {circuit.name}")
    print(f"Qubits: {circuit.num_qubits}")
    print(f"Depth: {circuit.depth}")
    print(f"Gate count: {circuit.gate_count}")
    
    # Simulate
    simulator = QuantumSimulator()
    result = simulator.run(circuit, shots=1000)
    
    print("\nMeasurement results (1000 shots):")
    for state, count in sorted(result.counts.items(), key=lambda x: -x[1]):
        print(f"  |{state}⟩: {count} ({count/10:.1f}%)")
    
    return circuit, result


def example_grover_search():
    """Run Grover's quantum search algorithm."""
    print("\n" + "=" * 60)
    print("Example 4: Grover's Search Algorithm")
    print("=" * 60)
    
    from quantumviz import QuantumAlgorithms
    
    # Search for state |5⟩ = |101⟩ in a 3-qubit space
    num_qubits = 3
    target = 5
    
    print(f"\nSearching for state |{target}⟩ = |{format(target, f'0{num_qubits}b')}⟩")
    print(f"Search space: {2**num_qubits} states")
    
    # Run Grover's algorithm
    result = QuantumAlgorithms.run_grover(
        num_qubits=num_qubits,
        target_states=[target],
        shots=1000
    )
    
    print(f"\nIterations used: {result.iterations}")
    print(f"Success probability: {result.success_probability:.2%}")
    print(f"Most likely result: |{result.classical_result}⟩")
    
    print("\nMeasurement results:")
    for state, count in sorted(result.simulation_result.counts.items(), key=lambda x: -x[1])[:5]:
        marker = " ← TARGET" if int(state, 2) == target else ""
        print(f"  |{state}⟩: {count} ({count/10:.1f}%){marker}")
    
    return result


def example_qft():
    """Run Quantum Fourier Transform."""
    print("\n" + "=" * 60)
    print("Example 5: Quantum Fourier Transform")
    print("=" * 60)
    
    from quantumviz import QuantumAlgorithms
    
    # Build QFT circuit
    circuit = QuantumAlgorithms.qft(4)
    
    print(f"\nQFT Circuit:")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Depth: {circuit.depth}")
    print(f"  Gates: {circuit.gate_count}")
    
    # Simulate
    simulator = QuantumSimulator()
    simulator.enable_history_capture(True)
    result = simulator.run(circuit, shots=1000)
    
    print("\nOutput state probabilities (top 5):")
    probs = result.final_state.get_probabilities()
    sorted_probs = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    for i, p in sorted_probs:
        print(f"  |{format(i, '04b')}⟩: {p:.4f}")
    
    return circuit


def example_visualization_themes():
    """Demonstrate different visualization themes."""
    print("\n" + "=" * 60)
    print("Example 6: Visualization Themes")
    print("=" * 60)
    
    import numpy as np
    
    # Create a state to visualize
    state = QuantumState.from_bloch(
        theta=np.pi / 3,
        phi=np.pi / 4,
        name="Example State"
    )
    
    themes = ["quantum_dark", "cyberpunk", "matrix", "aurora"]
    
    print("\nGenerating visualizations with different themes...")
    
    for theme_name in themes:
        theme = VisualizationTheme.get_theme(theme_name)
        renderer = BlochSphereRenderer(theme=theme)
        
        fig = renderer.render_state(state, title=f"Theme: {theme_name}")
        filename = f"bloch_{theme_name}.png"
        renderer.save(fig, filename)
        print(f"  Saved: {filename}")
    
    print("\nDone! Check the generated PNG files.")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("🔮 QuantumViz CLI - Examples")
    print("=" * 60)
    print("\nThis script demonstrates the main features of QuantumViz.")
    print("Each example shows different capabilities of the library.\n")
    
    # Run examples
    example_bell_state()
    example_ghz_state()
    example_custom_circuit()
    example_grover_search()
    example_qft()
    example_visualization_themes()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Try the CLI: quantumviz --help")
    print("  2. Use AI generation: quantumviz ai generate 'Bell state'")
    print("  3. Explore more: quantumviz visualize themes")
    print()


if __name__ == "__main__":
    main()
