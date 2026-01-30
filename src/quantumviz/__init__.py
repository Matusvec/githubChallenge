"""
🔮 QuantumViz CLI - Interactive 3D Quantum State Visualizer
============================================================

A powerful terminal tool that lets you simulate complex quantum systems
and generates stunning 3D visualizations right in your command line.

Features:
- Multi-qubit quantum simulations (up to 20 qubits)
- Interactive 3D Bloch sphere visualizations with 7 stunning themes
- Animated quantum state evolutions (GIF/MP4 export)
- AI-powered circuit generation via GitHub Copilot SDK
- Quantum algorithm visualization (Grover, QFT, VQE, Phase Estimation)
- Realistic noise modeling and decoherence simulation
- Export to PNG, SVG, GIF, QASM, LaTeX, and JSON
- Circuit optimization with ML-based techniques

Built for the GitHub Copilot SDK Challenge 2024

Copyright (c) 2024 QuantumViz Team
Licensed under the MIT License
"""

__version__ = "1.0.0"
__author__ = "QuantumViz Team"
__email__ = "quantum@example.com"

# Core imports
from quantumviz.core.quantum_state import QuantumState
from quantumviz.core.simulator import QuantumSimulator, QuantumCircuit
from quantumviz.core.gates import QuantumGates
from quantumviz.core.noise import NoiseModel
from quantumviz.core.algorithms import QuantumAlgorithms

# Visualization imports
from quantumviz.visualization.bloch import BlochSphereRenderer
from quantumviz.visualization.animation import QuantumAnimator
from quantumviz.visualization.circuit_drawer import CircuitDrawer
from quantumviz.visualization.themes import VisualizationTheme

# AI imports
from quantumviz.ai.copilot import CopilotCircuitGenerator
from quantumviz.ai.parser import NaturalLanguageParser
from quantumviz.ai.optimizer import AIOptimizer

# Utilities
from quantumviz.utils.export import ExportManager
from quantumviz.utils.config import Config

__all__ = [
    # Core
    "QuantumState",
    "QuantumSimulator",
    "QuantumCircuit",
    "QuantumGates",
    "NoiseModel",
    "QuantumAlgorithms",
    # Visualization
    "BlochSphereRenderer",
    "QuantumAnimator",
    "CircuitDrawer",
    "VisualizationTheme",
    # AI
    "CopilotCircuitGenerator",
    "NaturalLanguageParser",
    "AIOptimizer",
    # Utils
    "ExportManager",
    "Config",
    # Metadata
    "__version__",
    "__author__",
]


def get_version() -> str:
    """Return the current version."""
    return __version__


def quick_start():
    """Print quick start guide."""
    print("""
🔮 QuantumViz Quick Start
=========================

1. Create a simple Bell state:
   
   from quantumviz import QuantumState, BlochSphereRenderer
   
   state = QuantumState.bell_state('phi+')
   renderer = BlochSphereRenderer()
   renderer.render_multi_qubit(state)

2. Run Grover's algorithm:
   
   from quantumviz import QuantumAlgorithms
   
   result = QuantumAlgorithms.run_grover(num_qubits=3, target_states=[5])
   print(f"Found: {result.classical_result}")

3. AI-powered circuit generation:
   
   from quantumviz import CopilotCircuitGenerator
   
   gen = CopilotCircuitGenerator()
   circuit, desc = gen.generate_circuit("3-qubit GHZ state")

4. CLI usage:
   
   quantumviz simulate grover --qubits 3 --target 5
   quantumviz visualize bloch --state bell_phi+
   quantumviz ai generate "4-qubit QFT"

For more examples, visit: https://github.com/quantumviz/examples
    """)
