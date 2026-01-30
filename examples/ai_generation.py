"""
QuantumViz Examples: AI-Powered Circuit Generation
==================================================

This example demonstrates the GitHub Copilot SDK integration
for natural language quantum circuit generation.
"""

from quantumviz import (
    CopilotCircuitGenerator,
    NaturalLanguageParser,
    QuantumSimulator,
    BlochSphereRenderer,
    CircuitDrawer,
    VisualizationTheme,
)


def example_basic_generation():
    """Basic circuit generation from natural language."""
    print("=" * 60)
    print("Example 1: Basic Circuit Generation")
    print("=" * 60)
    
    # Initialize the generator
    generator = CopilotCircuitGenerator()
    
    # Check if Copilot SDK is available
    if generator.is_copilot_available:
        print("\n✅ GitHub Copilot SDK is available!")
    else:
        print("\n⚠️ GitHub Copilot SDK not available, using fallback parser")
    
    # Generate different circuits
    prompts = [
        "Create a Bell state",
        "3-qubit GHZ state",
        "Quantum Fourier Transform on 4 qubits",
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: \"{prompt}\"")
        print("-" * 40)
        
        circuit, description = generator.generate_circuit(prompt)
        
        print(f"📊 Description: {description}")
        print(f"   Qubits: {circuit.num_qubits}")
        print(f"   Depth: {circuit.depth}")
        print(f"   Gates: {circuit.gate_count}")


def example_grover_generation():
    """Generate Grover's search circuit from natural language."""
    print("\n" + "=" * 60)
    print("Example 2: Grover's Algorithm Generation")
    print("=" * 60)
    
    generator = CopilotCircuitGenerator()
    
    prompt = "4-qubit Grover search for state 7 with 2 iterations"
    print(f"\n📝 Prompt: \"{prompt}\"")
    
    circuit, description = generator.generate_circuit(prompt)
    
    print(f"\n📊 Generated: {description}")
    
    # Simulate the circuit
    simulator = QuantumSimulator()
    result = simulator.run(circuit, shots=1000)
    
    print("\n📈 Simulation results:")
    for state, count in sorted(result.counts.items(), key=lambda x: -x[1])[:5]:
        decimal = int(state, 2)
        marker = " ← TARGET" if decimal == 7 else ""
        print(f"   |{state}⟩ = |{decimal}⟩: {count/10:.1f}%{marker}")


def example_code_generation():
    """Generate code for different quantum frameworks."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Framework Code Generation")
    print("=" * 60)
    
    generator = CopilotCircuitGenerator()
    
    # Generate a simple circuit
    circuit, _ = generator.generate_circuit("Bell state with measurements")
    
    # Generate code for different frameworks
    frameworks = ["quantumviz", "qiskit", "cirq"]
    
    for framework in frameworks:
        print(f"\n📦 {framework.upper()} Code:")
        print("-" * 40)
        code = generator.generate_code(circuit, framework=framework)
        print(code)


def example_complex_circuit():
    """Generate and visualize a complex circuit."""
    print("\n" + "=" * 60)
    print("Example 4: Complex Circuit with Visualization")
    print("=" * 60)
    
    generator = CopilotCircuitGenerator()
    
    prompt = "Create a 3-qubit circuit with Hadamard gates, CNOT entanglement, and T gates"
    print(f"\n📝 Prompt: \"{prompt}\"")
    
    circuit, description = generator.generate_circuit(prompt)
    
    print(f"\n📊 Generated: {description}")
    
    # Draw the circuit
    theme = VisualizationTheme.get_theme("cyberpunk")
    drawer = CircuitDrawer(theme)
    
    print("\n📐 ASCII Circuit Diagram:")
    print(drawer.to_ascii(circuit))
    
    # Simulate
    simulator = QuantumSimulator()
    simulator.enable_history_capture(True)
    result = simulator.run(circuit, shots=1000)
    
    # Visualize final state
    renderer = BlochSphereRenderer(theme=theme)
    fig = renderer.render_multi_qubit(
        result.final_state, 
        title="AI-Generated Circuit Result"
    )
    renderer.save(fig, "ai_generated_result.png")
    print("\n💾 Saved visualization to 'ai_generated_result.png'")


def example_optimization_hints():
    """Get AI-powered optimization suggestions."""
    print("\n" + "=" * 60)
    print("Example 5: Circuit Optimization Hints")
    print("=" * 60)
    
    from quantumviz import QuantumCircuit
    import numpy as np
    
    # Create a circuit with some redundancy
    circuit = QuantumCircuit(2, name="Unoptimized")
    circuit.h(0)
    circuit.h(0)  # Redundant - H*H = I
    circuit.x(1)
    circuit.x(1)  # Redundant - X*X = I
    circuit.cnot(0, 1)
    circuit.rz(0, np.pi / 4)
    circuit.rz(0, np.pi / 4)  # Can be merged
    
    print(f"\nOriginal circuit: {circuit.gate_count} gates")
    
    generator = CopilotCircuitGenerator()
    hints = generator.get_optimization_hints(circuit)
    
    print("\n💡 Optimization hints:")
    for i, hint in enumerate(hints, 1):
        print(f"   {i}. {hint}")


def example_natural_language_parser():
    """Demonstrate the fallback natural language parser."""
    print("\n" + "=" * 60)
    print("Example 6: Natural Language Parser (Fallback)")
    print("=" * 60)
    
    parser = NaturalLanguageParser()
    
    # Test various prompts
    prompts = [
        "Create a Bell state",
        "5-qubit GHZ state",
        "Grover search for target 3 in 3 qubits",
        "QFT on 4 qubits",
        "random circuit with depth 5",
    ]
    
    print("\nParsing natural language prompts:")
    
    for prompt in prompts:
        circuit = parser.parse(prompt)
        explanation = parser.explain_circuit(circuit.name.lower().replace("_", " ").split()[0])
        
        print(f"\n📝 \"{prompt}\"")
        print(f"   → {circuit.name} ({circuit.num_qubits} qubits, {circuit.gate_count} gates)")
    
    # Show suggestions
    print("\n💭 Autocomplete suggestions for 'gr':")
    suggestions = parser.suggest_completions("gr")
    for s in suggestions:
        print(f"   • {s}")


def example_interactive_session():
    """Simulate an interactive AI session."""
    print("\n" + "=" * 60)
    print("Example 7: Interactive AI Session")
    print("=" * 60)
    
    generator = CopilotCircuitGenerator()
    
    # Conversation-like interaction
    conversation = [
        "Create a simple entangled state",
        "Now make it a 3-qubit GHZ state",
        "Add measurements to all qubits",
    ]
    
    print("\n🤖 Simulating interactive session:\n")
    
    for i, prompt in enumerate(conversation, 1):
        print(f"👤 User: {prompt}")
        
        circuit, description = generator.generate_circuit(prompt)
        
        print(f"🤖 Assistant: Created {description}")
        print(f"   Circuit has {circuit.num_qubits} qubits and {circuit.gate_count} gates.\n")


def main():
    """Run all AI examples."""
    print("\n" + "=" * 60)
    print("🤖 QuantumViz CLI - AI-Powered Circuit Generation")
    print("=" * 60)
    print("\nThis script demonstrates the GitHub Copilot SDK integration")
    print("for natural language quantum circuit generation.\n")
    
    # Run examples
    example_basic_generation()
    example_grover_generation()
    example_code_generation()
    example_complex_circuit()
    example_optimization_hints()
    example_natural_language_parser()
    example_interactive_session()
    
    print("\n" + "=" * 60)
    print("✅ All AI examples completed!")
    print("=" * 60)
    print("\nTry it yourself with the CLI:")
    print("  quantumviz ai generate \"Your quantum circuit description\"")
    print("  quantumviz ai explain grover")
    print()


if __name__ == "__main__":
    main()
