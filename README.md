# 🔮 QuantumViz CLI - Interactive 3D Quantum State Visualizer

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Copilot SDK](https://img.shields.io/badge/GitHub%20Copilot%20SDK-Powered-purple.svg)](https://github.com/features/copilot)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/DEV.to-Hackathon%202024-orange.svg)](https://dev.to/challenges/github)

**A powerful CLI tool for quantum computing simulations with stunning 3D visualizations and AI-powered circuit generation**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Gallery](#-gallery) • [API](#-api-reference)

</div>

---

## 🌟 What is QuantumViz?

QuantumViz CLI is an **interactive 3D quantum state visualizer** that brings quantum computing to your terminal. Simulate complex quantum systems, watch quantum states evolve on beautiful Bloch spheres, and let AI generate circuits from natural language—all from your command line.

### Built for the GitHub Copilot SDK Challenge

This project showcases the power of the **GitHub Copilot SDK** by enabling:
- 🤖 **Natural language → quantum circuits**: "Create a 3-qubit GHZ state with noise"
- 🧠 **AI-powered optimization**: ML-based circuit compression and parameter tuning
- 📚 **Intelligent explanations**: Get AI explanations of quantum algorithms
- ✨ **Smart suggestions**: Context-aware completions and recommendations

---

## ✨ Features

### 🔬 Quantum Simulation Engine
- **Multi-qubit simulations** up to 20 qubits
- **Statevector and density matrix** representations
- **Comprehensive gate library**: Pauli, Clifford, rotation, controlled, and custom gates
- **Realistic noise models**: Depolarizing, amplitude/phase damping, T1/T2, thermal relaxation
- **Popular algorithms**: Grover's search, QFT, VQE, Phase Estimation, quantum walks

### 🎨 Stunning Visualizations
- **3D Bloch sphere rendering** with trajectories and animations
- **7 beautiful themes**: Quantum Dark, Cyberpunk, Matrix, Aurora, Sunset, and more
- **Circuit diagrams** in matplotlib, ASCII art, or LaTeX
- **Animated GIF/MP4 exports** of quantum state evolution
- **Real-time terminal animations** for quick previews

### 🤖 AI Integration (GitHub Copilot SDK)
- **Natural language circuit generation**: Describe circuits in plain English
- **Intelligent code generation**: Export to QuantumViz, Qiskit, or Cirq
- **Circuit optimization hints**: AI-powered suggestions for improvement
- **Algorithm explanations**: Interactive learning support

### 📤 Export Capabilities
- **Images**: PNG, SVG with customizable DPI
- **Data**: JSON, CSV, NumPy formats
- **Circuits**: OpenQASM, LaTeX Qcircuit
- **Animations**: GIF, MP4

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/quantumviz-cli.git
cd quantumviz-cli

# Install with pip
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,torch]"
```

### Verify Installation

```bash
quantumviz --version
# QuantumViz CLI version 1.0.0
```

---

## 🚀 Quick Start

### 1. Your First Quantum Simulation

```bash
# Create and visualize a Bell state
quantumviz simulate bell --type phi+ --visualize

# Run Grover's search algorithm
quantumviz simulate grover --qubits 3 --target 5 --visualize

# Create a GHZ state with 4 qubits
quantumviz simulate ghz --qubits 4 --theme cyberpunk
```

### 2. AI-Powered Circuit Generation

```bash
# Generate a circuit from natural language
quantumviz ai generate "Create a 3-qubit quantum Fourier transform"

# Generate with visualization
quantumviz ai generate "4-qubit Grover search for state 7" --visualize

# Get algorithm explanations
quantumviz ai explain grover
```

### 3. Visualize Quantum States

```bash
# Bloch sphere visualization
quantumviz visualize bloch --state random --qubits 2 --theme aurora

# Circuit diagram
quantumviz visualize circuit --algorithm qft --qubits 4 --ascii

# List available themes
quantumviz visualize themes
```

### 4. Create Animations

```bash
# Animate Grover's algorithm
quantumviz animate evolution --algorithm grover --qubits 2 --export grover.gif

# Terminal ASCII animation
quantumviz animate evolution --algorithm bell --terminal
```

---

## 📖 Usage

### CLI Commands Overview

```
quantumviz
├── simulate    # Run quantum simulations
│   ├── grover  # Grover's search algorithm
│   ├── qft     # Quantum Fourier Transform
│   ├── bell    # Bell states
│   ├── ghz     # GHZ states
│   └── vqe     # Variational Quantum Eigensolver
├── visualize   # Create visualizations
│   ├── bloch   # Bloch sphere
│   ├── circuit # Circuit diagrams
│   └── themes  # List themes
├── ai          # AI-powered features
│   ├── generate # Generate circuits from text
│   └── explain  # Algorithm explanations
├── animate     # Create animations
│   └── evolution # State evolution
└── export      # Export data
    └── state   # Export quantum state
```

### Detailed Examples

#### Simulate Grover's Algorithm

```bash
quantumviz simulate grover \
    --qubits 4 \
    --target 11 \
    --iterations 3 \
    --noise moderate \
    --visualize \
    --theme quantum_dark \
    --export grover_result.png
```

#### AI Circuit Generation

```bash
# Simple generation
quantumviz ai generate "Bell state"

# Complex circuit with parameters
quantumviz ai generate "5-qubit QFT with inverse at the end"

# With noise simulation
quantumviz ai generate "Grover search for state 5 in 3 qubits with depolarizing noise"
```

#### VQE for Ground State

```bash
quantumviz simulate vqe \
    --qubits 2 \
    --layers 3 \
    --iterations 100 \
    --visualize
```

---

## 🐍 Python API

### Basic Usage

```python
from quantumviz import (
    QuantumState, 
    QuantumSimulator, 
    QuantumCircuit,
    BlochSphereRenderer,
    QuantumAlgorithms,
)

# Create a Bell state
bell = QuantumState.bell_state('phi+')
print(f"Entanglement: {bell.concurrence():.4f}")

# Build a custom circuit
circuit = QuantumCircuit(3, name="My Circuit")
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
circuit.measure_all()

# Simulate
sim = QuantumSimulator()
result = sim.run(circuit, shots=1000)
print(result.counts)

# Visualize
renderer = BlochSphereRenderer()
fig = renderer.render_multi_qubit(result.final_state)
renderer.show(fig)
```

### AI-Powered Generation

```python
from quantumviz import CopilotCircuitGenerator

# Initialize the generator
gen = CopilotCircuitGenerator()

# Generate circuit from natural language
circuit, description = gen.generate_circuit(
    "Create a 4-qubit GHZ state with measurements"
)

print(description)
# "GHZ state (4 qubits) - (|0000⟩ + |1111⟩)/√2"

# Get code for different frameworks
qiskit_code = gen.generate_code(circuit, framework="qiskit")
print(qiskit_code)
```

### Noise Simulation

```python
from quantumviz import NoiseModel, QuantumSimulator

# Create realistic noise model
noise = NoiseModel.from_backend("moderate")
noise.add_depolarizing(probability=0.01)
noise.add_thermal_relaxation(T1=50e-6, T2=30e-6, gate_time=50e-9)

# Simulate with noise
sim = QuantumSimulator(noise_model=noise)
result = sim.run(circuit, shots=10000)
```

### Visualization Themes

```python
from quantumviz import BlochSphereRenderer, VisualizationTheme

# Use cyberpunk theme
theme = VisualizationTheme.get_theme("cyberpunk")
renderer = BlochSphereRenderer(theme=theme)

# Render with trajectory
fig = renderer.render_trajectory(
    state_history,
    title="Quantum Evolution",
    show_labels=True,
)
renderer.save(fig, "evolution.png", dpi=300)
```

---

## 🎨 Gallery

### Available Themes

| Theme | Description |
|-------|-------------|
| `quantum_dark` | Dark theme with neon quantum colors (default) |
| `quantum_light` | Light theme for presentations |
| `cyberpunk` | Neon cyberpunk aesthetic |
| `matrix` | Matrix/hacker green style |
| `aurora` | Northern lights inspired |
| `sunset` | Warm gradient colors |
| `scientific` | Clean publication style |

### Sample Visualizations

```
┌──────────────────────────────────────────────────────────────┐
│                    Grover's Search (3 qubits)                │
│                                                              │
│  q0: ─[H]─────●────[H]─[X]────●────[X]─[H]─                 │
│               │               │                              │
│  q1: ─[H]────●┼───[H]─[X]───●┼───[X]─[H]─                  │
│              ││              ││                              │
│  q2: ─[H]───[X]──[H]─[X]───[X]──[X]─[H]─                   │
│                                                              │
│  Target: |101⟩    Iterations: 2    Success: 94.5%            │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Set default theme
export QUANTUMVIZ_THEME=cyberpunk

# Set default shots
export QUANTUMVIZ_DEFAULT_SHOTS=10000

# Disable Copilot SDK (use fallback parser)
export QUANTUMVIZ_DISABLE_COPILOT=1

# Set API key for Copilot SDK
export COPILOT_API_KEY=your_api_key
```

### Config File

Create `~/.quantumviz/config.json`:

```json
{
  "visualization": {
    "default_theme": "quantum_dark",
    "dpi": 150,
    "animation_fps": 30
  },
  "simulation": {
    "default_shots": 1000,
    "max_qubits": 20
  },
  "ai": {
    "enable_copilot": true,
    "model": "gpt-4",
    "temperature": 0.3
  }
}
```

---

## 🏗️ Project Structure

```
quantumviz/
├── src/quantumviz/
│   ├── __init__.py          # Package entry point
│   ├── cli/                  # CLI interface (Typer)
│   │   ├── __init__.py
│   │   └── main.py          # Command definitions
│   ├── core/                 # Quantum simulation engine
│   │   ├── __init__.py
│   │   ├── quantum_state.py # State representation
│   │   ├── gates.py         # Gate library
│   │   ├── noise.py         # Noise models
│   │   ├── simulator.py     # Circuit simulation
│   │   └── algorithms.py    # Quantum algorithms
│   ├── visualization/        # Rendering engine
│   │   ├── __init__.py
│   │   ├── bloch.py         # 3D Bloch sphere
│   │   ├── animation.py     # GIF/MP4 animations
│   │   ├── circuit_drawer.py # Circuit diagrams
│   │   └── themes.py        # Color themes
│   ├── ai/                   # AI integration
│   │   ├── __init__.py
│   │   ├── copilot.py       # GitHub Copilot SDK
│   │   ├── parser.py        # NL parsing fallback
│   │   └── optimizer.py     # ML optimization
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── export.py        # Export manager
│       ├── config.py        # Configuration
│       └── helpers.py       # Helper functions
├── examples/                 # Example scripts
├── tests/                    # Test suite
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

---

## 🧪 Development

### Setup Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/quantumviz-cli.git
cd quantumviz-cli
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/
isort src/

# Type checking
mypy src/quantumviz
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=quantumviz --cov-report=html

# Specific module
pytest tests/test_simulator.py -v
```

---

## 📚 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `QuantumState` | Represents quantum states with Bloch coordinates |
| `QuantumCircuit` | Circuit builder with fluent API |
| `QuantumSimulator` | Statevector and density matrix simulation |
| `QuantumGates` | Comprehensive gate library |
| `NoiseModel` | Kraus operator-based noise simulation |
| `QuantumAlgorithms` | Pre-built quantum algorithms |

### Visualization Classes

| Class | Description |
|-------|-------------|
| `BlochSphereRenderer` | 3D Bloch sphere visualization |
| `QuantumAnimator` | Animation generation |
| `CircuitDrawer` | Circuit diagram rendering |
| `VisualizationTheme` | Theme configuration |

### AI Classes

| Class | Description |
|-------|-------------|
| `CopilotCircuitGenerator` | GitHub Copilot SDK integration |
| `NaturalLanguageParser` | Fallback NL parser |
| `AIOptimizer` | ML-based circuit optimization |

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **GitHub Copilot SDK** for AI-powered features
- The quantum computing community for inspiration
- All contributors and testers

---

<div align="center">

**Made with ❤️ for the GitHub Copilot SDK Challenge 2024**

[⬆ Back to Top](#-quantumviz-cli---interactive-3d-quantum-state-visualizer)

</div>
