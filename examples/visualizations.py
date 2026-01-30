"""
QuantumViz Examples: Stunning Visualizations
=============================================

This example demonstrates the visualization capabilities
including 3D Bloch spheres, animations, and theming.
"""

import numpy as np
from quantumviz import (
    QuantumState,
    QuantumCircuit,
    QuantumSimulator,
    BlochSphereRenderer,
    QuantumAnimator,
    CircuitDrawer,
    VisualizationTheme,
    QuantumAlgorithms,
)


def example_bloch_sphere():
    """Demonstrate 3D Bloch sphere visualization."""
    print("=" * 60)
    print("Example 1: 3D Bloch Sphere")
    print("=" * 60)
    
    # Create various single-qubit states
    states = [
        ("|0⟩", QuantumState.from_basis("0")),
        ("|1⟩", QuantumState.from_basis("1")),
        ("|+⟩", QuantumState(1, state_vector=np.array([1, 1]) / np.sqrt(2))),
        ("|-⟩", QuantumState(1, state_vector=np.array([1, -1]) / np.sqrt(2))),
        ("|i⟩", QuantumState(1, state_vector=np.array([1, 1j]) / np.sqrt(2))),
        ("Random", QuantumState.random_state(1)),
    ]
    
    theme = VisualizationTheme.get_theme("quantum_dark")
    renderer = BlochSphereRenderer(theme=theme)
    
    print("\nGenerating Bloch sphere visualizations...")
    
    for name, state in states:
        bloch = state.bloch_coordinates(0)
        print(f"\n{name}:")
        print(f"  Bloch coordinates: ({bloch.x:.3f}, {bloch.y:.3f}, {bloch.z:.3f})")
        
        fig = renderer.render_state(state, title=f"State {name}")
        filename = f"bloch_{name.replace('|', '').replace('⟩', '')}.png"
        renderer.save(fig, filename)
        print(f"  Saved: {filename}")


def example_multi_qubit_visualization():
    """Visualize multi-qubit entangled states."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Qubit Visualization")
    print("=" * 60)
    
    # Create different entangled states
    states = [
        ("Bell |Φ+⟩", QuantumState.bell_state("phi+")),
        ("Bell |Ψ-⟩", QuantumState.bell_state("psi-")),
        ("GHZ (3)", QuantumState.ghz_state(3)),
        ("W (3)", QuantumState.w_state(3)),
    ]
    
    theme = VisualizationTheme.get_theme("cyberpunk")
    renderer = BlochSphereRenderer(theme=theme)
    
    print("\nGenerating multi-qubit visualizations...")
    
    for name, state in states:
        print(f"\n{name}:")
        print(f"  Qubits: {state.num_qubits}")
        print(f"  Concurrence: {state.concurrence():.4f}")
        
        fig = renderer.render_multi_qubit(state, title=name)
        filename = f"multi_{name.replace(' ', '_').replace('|', '').replace('⟩', '')}.png"
        renderer.save(fig, filename)
        print(f"  Saved: {filename}")


def example_state_trajectory():
    """Visualize quantum state evolution trajectory."""
    print("\n" + "=" * 60)
    print("Example 3: State Evolution Trajectory")
    print("=" * 60)
    
    # Simulate a circuit and capture history
    circuit = QuantumCircuit(1, name="Evolution")
    
    # Create a sequence of rotations
    steps = 20
    for i in range(steps):
        circuit.rx(0, np.pi / steps)
        circuit.ry(0, np.pi / (2 * steps))
    
    simulator = QuantumSimulator()
    simulator.enable_history_capture(True)
    result = simulator.run(circuit)
    
    print(f"\nCaptured {len(result.state_history)} states in evolution")
    
    # Render trajectory
    theme = VisualizationTheme.get_theme("aurora")
    renderer = BlochSphereRenderer(theme=theme)
    
    fig = renderer.render_trajectory(
        result.state_history,
        title="Quantum State Evolution",
        show_path=True
    )
    renderer.save(fig, "trajectory.png")
    print("Saved: trajectory.png")


def example_circuit_diagrams():
    """Generate circuit diagrams in different formats."""
    print("\n" + "=" * 60)
    print("Example 4: Circuit Diagrams")
    print("=" * 60)
    
    # Build a sample circuit
    circuit = QuantumCircuit(3, name="Sample Circuit")
    circuit.h(0)
    circuit.h(1)
    circuit.cnot(0, 2)
    circuit.cnot(1, 2)
    circuit.t(2)
    circuit.s(0)
    circuit.cz(0, 1)
    circuit.measure_all()
    
    theme = VisualizationTheme.get_theme("scientific")
    drawer = CircuitDrawer(theme)
    
    # ASCII format
    print("\n📐 ASCII Circuit Diagram:")
    print(drawer.to_ascii(circuit))
    
    # LaTeX format
    print("\n📄 LaTeX Qcircuit:")
    print(drawer.to_latex(circuit))
    
    # Matplotlib figure
    fig = drawer.draw(circuit, show=False)
    drawer.save(fig, "circuit_diagram.png")
    print("\n💾 Saved: circuit_diagram.png")


def example_algorithm_visualization():
    """Visualize quantum algorithm execution."""
    print("\n" + "=" * 60)
    print("Example 5: Algorithm Visualization")
    print("=" * 60)
    
    # Grover's algorithm
    print("\n🔍 Grover's Search Algorithm")
    
    circuit, num_iterations = QuantumAlgorithms.grover_search(3, [5])
    
    theme = VisualizationTheme.get_theme("matrix")
    drawer = CircuitDrawer(theme)
    
    print(f"\nTarget: |5⟩ = |101⟩")
    print(f"Iterations: {num_iterations}")
    print(f"\nCircuit:")
    print(drawer.to_ascii(circuit))
    
    # Simulate and visualize result
    simulator = QuantumSimulator()
    simulator.enable_history_capture(True)
    result = simulator.run(circuit, shots=1000)
    
    renderer = BlochSphereRenderer(theme=theme)
    fig = renderer.render_multi_qubit(
        result.final_state,
        title="Grover's Search Final State"
    )
    renderer.save(fig, "grover_final.png")
    print("\nSaved: grover_final.png")


def example_animation():
    """Create animations of quantum state evolution."""
    print("\n" + "=" * 60)
    print("Example 6: State Evolution Animation")
    print("=" * 60)
    
    # Create circuit with evolution
    circuit = QuantumCircuit(1, name="Rotation")
    
    # Smooth rotation
    for i in range(30):
        circuit.ry(0, np.pi / 15)
    
    # Simulate
    simulator = QuantumSimulator()
    simulator.enable_history_capture(True)
    result = simulator.run(circuit)
    
    print(f"\nCaptured {len(result.state_history)} frames")
    
    # Create animator
    theme = VisualizationTheme.get_theme("sunset")
    animator = QuantumAnimator(theme=theme)
    
    # Generate animation
    print("Generating animation (this may take a moment)...")
    animator.animate_bloch_sphere(
        result.state_history,
        output_path="rotation_animation.gif",
        fps=10,
    )
    print("Saved: rotation_animation.gif")


def example_theme_gallery():
    """Create a gallery of all available themes."""
    print("\n" + "=" * 60)
    print("Example 7: Theme Gallery")
    print("=" * 60)
    
    from quantumviz.visualization.themes import ThemeName
    
    # Create a sample state
    state = QuantumState.from_bloch(theta=np.pi/3, phi=np.pi/4, name="Sample")
    
    themes = list(ThemeName)
    print(f"\nGenerating gallery with {len(themes)} themes...")
    
    for theme_name in themes:
        theme = VisualizationTheme.get_theme(theme_name.value)
        renderer = BlochSphereRenderer(theme=theme)
        
        fig = renderer.render_state(
            state, 
            title=f"Theme: {theme_name.value}"
        )
        
        filename = f"theme_{theme_name.value}.png"
        renderer.save(fig, filename)
        print(f"  {theme_name.value}: {filename}")
    
    print("\nTheme gallery complete!")


def example_probability_histogram():
    """Visualize measurement probability distributions."""
    print("\n" + "=" * 60)
    print("Example 8: Probability Histogram")
    print("=" * 60)
    
    # Create a circuit with interesting probability distribution
    circuit = QuantumCircuit(3, name="Distribution")
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)
    circuit.ccx(0, 1, 2)  # Toffoli
    circuit.measure_all()
    
    simulator = QuantumSimulator()
    result = simulator.run(circuit, shots=1000)
    
    print("\nMeasurement Distribution:")
    print("-" * 40)
    
    # Create ASCII histogram
    total = sum(result.counts.values())
    max_count = max(result.counts.values())
    
    for state in sorted(result.counts.keys()):
        count = result.counts[state]
        bar_len = int(count / max_count * 30)
        bar = "█" * bar_len
        prob = count / total
        print(f"|{state}⟩: {bar} {prob:.1%}")


def main():
    """Run all visualization examples."""
    print("\n" + "=" * 60)
    print("🎨 QuantumViz CLI - Visualization Examples")
    print("=" * 60)
    print("\nThis script demonstrates the stunning visualization")
    print("capabilities of QuantumViz CLI.\n")
    
    # Run examples
    example_bloch_sphere()
    example_multi_qubit_visualization()
    example_state_trajectory()
    example_circuit_diagrams()
    example_algorithm_visualization()
    example_animation()
    example_theme_gallery()
    example_probability_histogram()
    
    print("\n" + "=" * 60)
    print("✅ All visualization examples completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - bloch_*.png: Single qubit Bloch spheres")
    print("  - multi_*.png: Multi-qubit visualizations")
    print("  - trajectory.png: State evolution trajectory")
    print("  - circuit_diagram.png: Circuit diagram")
    print("  - grover_final.png: Grover algorithm result")
    print("  - rotation_animation.gif: Animated evolution")
    print("  - theme_*.png: Theme gallery")
    print()


if __name__ == "__main__":
    main()
