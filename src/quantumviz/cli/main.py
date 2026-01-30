"""
QuantumViz CLI - Main Entry Point
==================================

🔮 Interactive 3D Quantum State Visualizer with AI-Powered Simulations

Usage:
    quantumviz simulate grover --qubits 3 --target 5
    quantumviz visualize state --type bell --theme cyberpunk
    quantumviz ai generate "Create a 4-qubit QFT circuit"
    quantumviz animate evolution --algorithm vqe --export animation.gif
"""

from __future__ import annotations
import typer
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from pathlib import Path
import sys

# Initialize Typer app
app = typer.Typer(
    name="quantumviz",
    help="🔮 QuantumViz CLI - Interactive 3D Quantum State Visualizer",
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)

# Console for rich output
console = Console()

# Sub-command groups
simulate_app = typer.Typer(help="🧪 Run quantum simulations")
visualize_app = typer.Typer(help="🎨 Create stunning visualizations")
ai_app = typer.Typer(help="🤖 AI-powered circuit generation with GitHub Copilot SDK")
animate_app = typer.Typer(help="🎬 Generate animations")
export_app = typer.Typer(help="📤 Export results")

app.add_typer(simulate_app, name="simulate")
app.add_typer(visualize_app, name="visualize")
app.add_typer(ai_app, name="ai")
app.add_typer(animate_app, name="animate")
app.add_typer(export_app, name="export")


def print_banner():
    """Print the QuantumViz banner."""
    banner = """
[bold cyan]
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🔮 [bold magenta]Q U A N T U M V I Z[/bold magenta]   C L I                        ║
    ║                                                               ║
    ║   Interactive 3D Quantum State Visualizer                     ║
    ║   Powered by GitHub Copilot SDK                               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
[/bold cyan]
    """
    console.print(banner)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
):
    """
    🔮 QuantumViz CLI - Interactive 3D Quantum State Visualizer
    
    Simulate complex quantum systems and generate stunning visualizations
    right from your terminal. AI-powered circuit generation with GitHub Copilot SDK.
    
    Examples:
    
        quantumviz simulate grover --qubits 3 --target 5
        
        quantumviz visualize bloch --state bell_phi+
        
        quantumviz ai generate "3-qubit Grover search with noise"
    """
    if version:
        from quantumviz import __version__
        console.print(f"[bold cyan]QuantumViz CLI[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()
    
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("\n[dim]Run 'quantumviz --help' for usage information.[/dim]\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@simulate_app.command("grover")
def simulate_grover(
    qubits: int = typer.Option(3, "--qubits", "-q", help="Number of qubits"),
    target: int = typer.Option(5, "--target", "-t", help="Target state to search for"),
    iterations: Optional[int] = typer.Option(None, "--iterations", "-i", help="Number of Grover iterations"),
    shots: int = typer.Option(1000, "--shots", "-s", help="Number of measurement shots"),
    noise: str = typer.Option("ideal", "--noise", "-n", help="Noise model: ideal, low, moderate, high"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    export_path: Optional[str] = typer.Option(None, "--export", "-e", help="Export visualization path"),
    theme: str = typer.Option("quantum_dark", "--theme", help="Visualization theme"),
):
    """
    🔍 Run Grover's quantum search algorithm.
    
    Searches for a target state in an unsorted database with quadratic speedup.
    
    Examples:
    
        quantumviz simulate grover --qubits 4 --target 7 --visualize
        
        quantumviz simulate grover -q 3 -t 5 --noise moderate --export grover.png
    """
    from quantumviz.core.algorithms import QuantumAlgorithms
    from quantumviz.core.noise import NoiseModel
    from quantumviz.visualization.bloch import BlochSphereRenderer
    from quantumviz.visualization.themes import VisualizationTheme
    
    console.print(Panel.fit(
        f"[bold cyan]🔍 Grover's Search Algorithm[/bold cyan]\n\n"
        f"Qubits: [green]{qubits}[/green]\n"
        f"Target: [yellow]{target}[/yellow] (binary: {format(target, f'0{qubits}b')})\n"
        f"Search space: [magenta]{2**qubits}[/magenta] states",
        title="Configuration"
    ))
    
    # Validate target
    if target >= 2 ** qubits:
        console.print(f"[red]Error: Target {target} is out of range for {qubits} qubits (max: {2**qubits - 1})[/red]")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running Grover's algorithm...", total=None)
        
        # Set up noise model
        noise_model = None
        if noise != "ideal":
            noise_model = NoiseModel.noisy_simulator(noise)
        
        # Run algorithm
        result = QuantumAlgorithms.run_grover(
            num_qubits=qubits,
            target_states=[target],
            shots=shots,
            noise_model=noise_model
        )
        
        progress.update(task, description="[green]✓ Simulation complete!")
    
    # Display results
    _display_algorithm_results(result, qubits)
    
    # Visualize if requested
    if visualize and result.simulation_result.state_history:
        console.print("\n[cyan]Generating visualization...[/cyan]")
        
        viz_theme = VisualizationTheme.get_theme(theme)
        renderer = BlochSphereRenderer(theme=viz_theme)
        
        if qubits <= 2:
            fig = renderer.render_trajectory(
                result.simulation_result.state_history,
                title=f"Grover's Search Evolution ({qubits} qubits)"
            )
        else:
            # For multi-qubit, show final state
            fig = renderer.render_multi_qubit(
                result.simulation_result.final_state,
                title=f"Grover's Search Final State"
            )
        
        if export_path:
            renderer.save(fig, export_path)
            console.print(f"[green]✓ Saved to {export_path}[/green]")
        else:
            renderer.show(fig)


@simulate_app.command("qft")
def simulate_qft(
    qubits: int = typer.Option(3, "--qubits", "-q", help="Number of qubits"),
    input_state: Optional[str] = typer.Option(None, "--input", "-i", help="Initial state (binary string)"),
    inverse: bool = typer.Option(False, "--inverse", help="Run inverse QFT"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    theme: str = typer.Option("quantum_dark", "--theme", help="Visualization theme"),
):
    """
    📊 Run Quantum Fourier Transform.
    
    The quantum analog of the discrete Fourier transform.
    
    Examples:
    
        quantumviz simulate qft --qubits 4
        
        quantumviz simulate qft -q 3 --input 101
    """
    from quantumviz.core.algorithms import QuantumAlgorithms
    from quantumviz.core.simulator import QuantumSimulator
    from quantumviz.visualization.circuit_drawer import CircuitDrawer
    from quantumviz.visualization.themes import VisualizationTheme
    
    console.print(Panel.fit(
        f"[bold cyan]📊 Quantum Fourier Transform[/bold cyan]\n\n"
        f"Qubits: [green]{qubits}[/green]\n"
        f"Input: [yellow]{input_state or '|0...0⟩'}[/yellow]\n"
        f"Inverse: [magenta]{inverse}[/magenta]",
        title="Configuration"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Building QFT circuit...", total=None)
        
        circuit = QuantumAlgorithms.qft(qubits, inverse=inverse)
        
        progress.update(task, description="[cyan]Running simulation...")
        
        simulator = QuantumSimulator()
        simulator.enable_history_capture(True)
        
        # Prepare initial state if provided
        initial_state = None
        if input_state:
            from quantumviz.core.quantum_state import QuantumState
            initial_state = QuantumState.from_basis(input_state)
        
        result = simulator.run(circuit, initial_state=initial_state, shots=1000)
        
        progress.update(task, description="[green]✓ Complete!")
    
    # Display circuit
    drawer = CircuitDrawer(VisualizationTheme.get_theme(theme))
    console.print("\n[bold]Circuit:[/bold]")
    console.print(drawer.to_ascii(circuit))
    
    # Display probabilities
    console.print("\n[bold]Output Probabilities:[/bold]")
    probs = result.get_probabilities()
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("State", style="yellow")
    table.add_column("Probability", style="green")
    table.add_column("Bar", style="magenta")
    
    for state, prob in sorted(probs.items(), key=lambda x: -x[1])[:8]:
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        table.add_row(f"|{state}⟩", f"{prob:.4f}", bar)
    
    console.print(table)
    
    if visualize:
        fig = drawer.draw(circuit, show=True)


@simulate_app.command("bell")
def simulate_bell(
    state_type: str = typer.Option("phi+", "--type", "-t", help="Bell state type: phi+, phi-, psi+, psi-"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show Bloch sphere"),
    theme: str = typer.Option("quantum_dark", "--theme", help="Visualization theme"),
):
    """
    🔔 Create and visualize Bell states (maximally entangled).
    
    Bell states are the simplest entangled quantum states.
    
    Examples:
    
        quantumviz simulate bell --type phi+
        
        quantumviz simulate bell -t psi- --theme cyberpunk
    """
    from quantumviz.core.quantum_state import QuantumState
    from quantumviz.visualization.bloch import BlochSphereRenderer
    from quantumviz.visualization.themes import VisualizationTheme
    
    console.print(Panel.fit(
        f"[bold cyan]🔔 Bell State: |{state_type}⟩[/bold cyan]\n\n"
        f"Maximally entangled 2-qubit state",
        title="Bell State"
    ))
    
    state = QuantumState.bell_state(state_type)
    
    # Display state info
    console.print(f"\n{state}\n")
    console.print(f"[bold]Concurrence:[/bold] [green]{state.concurrence():.4f}[/green] (1.0 = maximally entangled)")
    console.print(f"[bold]Entropy:[/bold] [yellow]{state.entropy():.4f}[/yellow]")
    
    if visualize:
        viz_theme = VisualizationTheme.get_theme(theme)
        renderer = BlochSphereRenderer(theme=viz_theme)
        
        fig = renderer.render_multi_qubit(state, title=f"Bell State |{state_type}⟩")
        renderer.show(fig)


@simulate_app.command("ghz")
def simulate_ghz(
    qubits: int = typer.Option(3, "--qubits", "-q", help="Number of qubits"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    theme: str = typer.Option("quantum_dark", "--theme", help="Visualization theme"),
):
    """
    🌟 Create GHZ (Greenberger-Horne-Zeilinger) state.
    
    Multi-qubit entangled state: (|00...0⟩ + |11...1⟩)/√2
    
    Examples:
    
        quantumviz simulate ghz --qubits 4
        
        quantumviz simulate ghz -q 5 --theme aurora
    """
    from quantumviz.core.quantum_state import QuantumState
    from quantumviz.core.algorithms import QuantumAlgorithms
    from quantumviz.core.simulator import QuantumSimulator
    from quantumviz.visualization.bloch import BlochSphereRenderer
    from quantumviz.visualization.themes import VisualizationTheme
    
    console.print(Panel.fit(
        f"[bold cyan]🌟 GHZ State ({qubits} qubits)[/bold cyan]\n\n"
        f"(|{'0'*qubits}⟩ + |{'1'*qubits}⟩) / √2",
        title="GHZ State"
    ))
    
    # Build and run circuit
    circuit = QuantumAlgorithms.ghz_state_circuit(qubits)
    
    simulator = QuantumSimulator()
    simulator.enable_history_capture(True)
    result = simulator.run(circuit, shots=1000)
    
    # Display results
    console.print(f"\n{result.final_state}\n")
    
    # Measurement results
    if result.counts:
        console.print("[bold]Measurement Results:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("State", style="yellow")
        table.add_column("Counts", style="green")
        table.add_column("Probability", style="magenta")
        
        for state, count in sorted(result.counts.items(), key=lambda x: -x[1]):
            prob = count / sum(result.counts.values())
            table.add_row(f"|{state}⟩", str(count), f"{prob:.2%}")
        
        console.print(table)
    
    if visualize:
        viz_theme = VisualizationTheme.get_theme(theme)
        renderer = BlochSphereRenderer(theme=viz_theme)
        
        fig = renderer.render_multi_qubit(result.final_state, title=f"GHZ State ({qubits} qubits)")
        renderer.show(fig)


@simulate_app.command("vqe")
def simulate_vqe(
    qubits: int = typer.Option(2, "--qubits", "-q", help="Number of qubits"),
    layers: int = typer.Option(2, "--layers", "-l", help="Ansatz layers"),
    iterations: int = typer.Option(50, "--iterations", "-i", help="Max optimization iterations"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show optimization progress"),
    theme: str = typer.Option("quantum_dark", "--theme", help="Visualization theme"),
):
    """
    ⚡ Run Variational Quantum Eigensolver.
    
    Hybrid quantum-classical algorithm to find ground state energy.
    
    Examples:
    
        quantumviz simulate vqe --qubits 2 --layers 3
        
        quantumviz simulate vqe -q 2 -l 2 -i 100
    """
    import numpy as np
    from quantumviz.core.algorithms import QuantumAlgorithms
    
    console.print(Panel.fit(
        f"[bold cyan]⚡ Variational Quantum Eigensolver[/bold cyan]\n\n"
        f"Qubits: [green]{qubits}[/green]\n"
        f"Ansatz Layers: [yellow]{layers}[/yellow]\n"
        f"Max Iterations: [magenta]{iterations}[/magenta]",
        title="VQE Configuration"
    ))
    
    # Create a simple Hamiltonian (e.g., Ising model)
    # H = -Σ Z_i Z_{i+1} - Σ X_i
    dim = 2 ** qubits
    H = np.zeros((dim, dim), dtype=complex)
    
    # Simple transverse field Ising Hamiltonian
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # Build Hamiltonian
    for i in range(qubits):
        # X term
        term = np.eye(1, dtype=complex)
        for j in range(qubits):
            term = np.kron(term, X if i == j else I)
        H -= 0.5 * term
        
        # ZZ term
        if i < qubits - 1:
            term = np.eye(1, dtype=complex)
            for j in range(qubits):
                if j == i or j == i + 1:
                    term = np.kron(term, Z)
                else:
                    term = np.kron(term, I)
            H -= term
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Optimizing VQE...", total=None)
        
        result = QuantumAlgorithms.run_vqe(
            hamiltonian=H,
            num_qubits=qubits,
            layers=layers,
            max_iterations=iterations
        )
        
        progress.update(task, description="[green]✓ Optimization complete!")
    
    # Display results
    console.print(f"\n[bold]Ground State Energy Estimate:[/bold] [green]{result.classical_result:.6f}[/green]")
    console.print(f"[bold]Iterations:[/bold] [yellow]{result.iterations}[/yellow]")
    
    # Exact ground state for comparison
    eigenvalues = np.linalg.eigvalsh(H)
    exact_ground = eigenvalues[0]
    console.print(f"[bold]Exact Ground State Energy:[/bold] [cyan]{exact_ground:.6f}[/cyan]")
    console.print(f"[bold]Error:[/bold] [magenta]{abs(result.classical_result - exact_ground):.6f}[/magenta]")
    
    if visualize and result.parameters and 'energy_history' in result.parameters:
        import matplotlib.pyplot as plt
        from quantumviz.visualization.themes import VisualizationTheme
        
        theme_obj = VisualizationTheme.get_theme(theme)
        plt.style.use('dark_background' if 'dark' in theme else 'default')
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=theme_obj.colors.background)
        ax.set_facecolor(theme_obj.colors.surface)
        
        energies = result.parameters['energy_history']
        ax.plot(energies, color=theme_obj.colors.primary, linewidth=2, label='VQE Energy')
        ax.axhline(y=exact_ground, color=theme_obj.colors.secondary, linestyle='--', 
                  label=f'Exact ({exact_ground:.4f})')
        
        ax.set_xlabel('Iteration', color=theme_obj.colors.text)
        ax.set_ylabel('Energy', color=theme_obj.colors.text)
        ax.set_title('VQE Optimization Progress', color=theme_obj.colors.text)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@visualize_app.command("bloch")
def visualize_bloch(
    state_type: str = typer.Option("random", "--state", "-s", help="State type: random, plus, minus, zero, one, bell_phi+, etc."),
    qubits: int = typer.Option(1, "--qubits", "-q", help="Number of qubits (for random states)"),
    theme: str = typer.Option("quantum_dark", "--theme", "-t", help="Visualization theme"),
    export_path: Optional[str] = typer.Option(None, "--export", "-e", help="Export path (PNG/SVG)"),
    ascii_mode: bool = typer.Option(False, "--ascii", "-a", help="ASCII art mode for terminal"),
):
    """
    🌐 Visualize quantum state on Bloch sphere.
    
    Examples:
    
        quantumviz visualize bloch --state random --theme cyberpunk
        
        quantumviz visualize bloch -s bell_phi+ --export bloch.png
        
        quantumviz visualize bloch --ascii
    """
    from quantumviz.core.quantum_state import QuantumState
    from quantumviz.visualization.bloch import BlochSphereRenderer
    from quantumviz.visualization.themes import VisualizationTheme
    import numpy as np
    
    # Create state based on type
    if state_type == "random":
        state = QuantumState.random_state(qubits, name="random")
    elif state_type == "zero" or state_type == "0":
        state = QuantumState.from_basis("0" * qubits, name="|0⟩")
    elif state_type == "one" or state_type == "1":
        state = QuantumState.from_basis("1" * qubits, name="|1⟩")
    elif state_type == "plus":
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        state = QuantumState(num_qubits=1, state_vector=sv, name="|+⟩")
    elif state_type == "minus":
        sv = np.array([1, -1], dtype=complex) / np.sqrt(2)
        state = QuantumState(num_qubits=1, state_vector=sv, name="|-⟩")
    elif state_type.startswith("bell_"):
        bell_type = state_type.replace("bell_", "")
        state = QuantumState.bell_state(bell_type)
    elif state_type.startswith("ghz"):
        n = int(state_type.replace("ghz", "") or "3")
        state = QuantumState.ghz_state(n)
    else:
        console.print(f"[red]Unknown state type: {state_type}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold cyan]🌐 Bloch Sphere Visualization[/bold cyan]\n\n"
        f"State: [yellow]{state.name}[/yellow]\n"
        f"Qubits: [green]{state.num_qubits}[/green]\n"
        f"Type: [magenta]{state.state_type.value}[/magenta]",
        title="Configuration"
    ))
    
    viz_theme = VisualizationTheme.get_theme(theme)
    renderer = BlochSphereRenderer(theme=viz_theme)
    
    if ascii_mode:
        console.print("\n" + renderer.to_ascii(state))
    else:
        if state.num_qubits == 1:
            fig = renderer.render_state(state, title=f"Bloch Sphere: {state.name}")
        else:
            fig = renderer.render_multi_qubit(state, title=f"Multi-Qubit State: {state.name}")
        
        if export_path:
            renderer.save(fig, export_path)
            console.print(f"[green]✓ Saved to {export_path}[/green]")
        else:
            renderer.show(fig)


@visualize_app.command("circuit")
def visualize_circuit(
    algorithm: str = typer.Option("bell", "--algorithm", "-a", help="Algorithm: bell, ghz, qft, grover"),
    qubits: int = typer.Option(3, "--qubits", "-q", help="Number of qubits"),
    theme: str = typer.Option("quantum_dark", "--theme", "-t", help="Visualization theme"),
    export_path: Optional[str] = typer.Option(None, "--export", "-e", help="Export path"),
    ascii_mode: bool = typer.Option(False, "--ascii", help="ASCII art mode"),
    latex_mode: bool = typer.Option(False, "--latex", help="Output LaTeX Qcircuit format"),
):
    """
    📐 Visualize quantum circuit diagrams.
    
    Examples:
    
        quantumviz visualize circuit --algorithm qft --qubits 4
        
        quantumviz visualize circuit -a grover -q 3 --ascii
        
        quantumviz visualize circuit --algorithm bell --latex
    """
    from quantumviz.core.algorithms import QuantumAlgorithms
    from quantumviz.visualization.circuit_drawer import CircuitDrawer
    from quantumviz.visualization.themes import VisualizationTheme
    
    # Build circuit based on algorithm
    if algorithm == "bell":
        circuit = QuantumAlgorithms.bell_state_circuit()
    elif algorithm == "ghz":
        circuit = QuantumAlgorithms.ghz_state_circuit(qubits)
    elif algorithm == "qft":
        circuit = QuantumAlgorithms.qft(qubits)
    elif algorithm == "grover":
        circuit, _ = QuantumAlgorithms.grover_search(qubits, [1])
    elif algorithm == "w":
        circuit = QuantumAlgorithms.w_state_circuit(qubits)
    else:
        console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold cyan]📐 Circuit: {algorithm.upper()}[/bold cyan]\n\n"
        f"Qubits: [green]{circuit.num_qubits}[/green]\n"
        f"Depth: [yellow]{circuit.depth}[/yellow]\n"
        f"Gates: [magenta]{circuit.gate_count}[/magenta]",
        title="Circuit Info"
    ))
    
    viz_theme = VisualizationTheme.get_theme(theme)
    drawer = CircuitDrawer(theme=viz_theme)
    
    if ascii_mode:
        console.print("\n" + drawer.to_ascii(circuit))
    elif latex_mode:
        console.print("\n[bold]LaTeX Qcircuit:[/bold]\n")
        console.print(drawer.to_latex(circuit))
    else:
        fig = drawer.draw(circuit, output_path=export_path, show=export_path is None)
        if export_path:
            console.print(f"[green]✓ Saved to {export_path}[/green]")


@visualize_app.command("themes")
def list_themes():
    """
    🎨 List available visualization themes.
    """
    from quantumviz.visualization.themes import VisualizationTheme, ThemeName
    
    console.print("\n[bold cyan]🎨 Available Themes[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Theme", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Primary Color", style="green")
    
    descriptions = {
        "quantum_dark": "Dark theme with quantum-inspired neon colors",
        "quantum_light": "Light theme for presentations",
        "cyberpunk": "Neon cyberpunk aesthetic with glow effects",
        "scientific": "Clean scientific publication style",
        "matrix": "Matrix/hacker green aesthetic",
        "aurora": "Northern lights inspired colors",
        "sunset": "Warm sunset gradient colors",
    }
    
    for name in ThemeName:
        theme = VisualizationTheme.get_theme(name.value)
        desc = descriptions.get(name.value, "")
        table.add_row(name.value, desc, theme.colors.primary)
    
    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# AI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@ai_app.command("generate")
def ai_generate(
    prompt: str = typer.Argument(..., help="Natural language description of the quantum circuit"),
    execute: bool = typer.Option(True, "--execute/--no-execute", help="Execute the generated circuit"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    export_path: Optional[str] = typer.Option(None, "--export", "-e", help="Export path"),
    theme: str = typer.Option("quantum_dark", "--theme", help="Visualization theme"),
):
    """
    🤖 Generate quantum circuits using AI (GitHub Copilot SDK).
    
    Describe what you want in natural language and let AI create the circuit.
    
    Examples:
    
        quantumviz ai generate "Create a 3-qubit GHZ state"
        
        quantumviz ai generate "4-qubit Grover search for state 7 with 2 iterations"
        
        quantumviz ai generate "Bell state followed by measurement"
    """
    console.print(Panel.fit(
        f"[bold cyan]🤖 AI Circuit Generation[/bold cyan]\n\n"
        f"Prompt: [yellow]{prompt}[/yellow]",
        title="GitHub Copilot SDK"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating circuit with Copilot SDK...", total=None)
        
        try:
            from quantumviz.ai.copilot import CopilotCircuitGenerator
            
            generator = CopilotCircuitGenerator()
            circuit, description = generator.generate_circuit(prompt)
            
            progress.update(task, description="[green]✓ Circuit generated!")
            
        except ImportError as e:
            progress.update(task, description="[yellow]⚠ Using fallback parser...")
            
            from quantumviz.ai.parser import NaturalLanguageParser
            parser = NaturalLanguageParser()
            circuit = parser.parse(prompt)
            description = f"Parsed circuit from: {prompt}"
    
    console.print(f"\n[bold]Generated Circuit:[/bold]")
    console.print(f"[dim]{description}[/dim]\n")
    
    # Show circuit
    from quantumviz.visualization.circuit_drawer import CircuitDrawer
    from quantumviz.visualization.themes import VisualizationTheme
    
    drawer = CircuitDrawer(VisualizationTheme.get_theme(theme))
    console.print(drawer.to_ascii(circuit))
    
    if execute:
        console.print("\n[cyan]Executing circuit...[/cyan]")
        
        from quantumviz.core.simulator import QuantumSimulator
        simulator = QuantumSimulator()
        simulator.enable_history_capture(True)
        result = simulator.run(circuit, shots=1000)
        
        # Show results
        if result.counts:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("State", style="yellow")
            table.add_column("Probability", style="green")
            
            for state, count in sorted(result.counts.items(), key=lambda x: -x[1])[:8]:
                prob = count / 1000
                table.add_row(f"|{state}⟩", f"{prob:.2%}")
            
            console.print("\n[bold]Results:[/bold]")
            console.print(table)
        
        if visualize:
            from quantumviz.visualization.bloch import BlochSphereRenderer
            
            renderer = BlochSphereRenderer(VisualizationTheme.get_theme(theme))
            
            if circuit.num_qubits <= 4:
                fig = renderer.render_multi_qubit(result.final_state, 
                                                  title="AI-Generated Circuit Result")
                if export_path:
                    renderer.save(fig, export_path)
                    console.print(f"[green]✓ Saved to {export_path}[/green]")
                else:
                    renderer.show(fig)


@ai_app.command("explain")
def ai_explain(
    algorithm: str = typer.Argument(..., help="Algorithm to explain: grover, qft, vqe, bell, ghz"),
):
    """
    📚 Get AI-powered explanation of quantum algorithms.
    
    Examples:
    
        quantumviz ai explain grover
        
        quantumviz ai explain vqe
    """
    explanations = {
        "grover": """
[bold cyan]🔍 Grover's Search Algorithm[/bold cyan]

[yellow]What it does:[/yellow]
Searches an unsorted database of N items in O(√N) time, compared to O(N) classically.
This is a quadratic speedup!

[yellow]Key concepts:[/yellow]
1. [green]Oracle:[/green] Marks the target state with a phase flip (-1)
2. [green]Diffusion:[/green] Amplifies probability of marked states (inversion about mean)
3. [green]Iterations:[/green] Repeat √N times for optimal success probability

[yellow]Circuit structure:[/yellow]
• Initialize: Apply H to all qubits (create superposition)
• Repeat O(√N) times:
  - Apply Oracle (phase flip targets)
  - Apply Diffusion (amplitude amplification)
• Measure

[yellow]Why it's important:[/yellow]
Demonstrates quantum speedup for unstructured search problems.
Foundation for more complex quantum algorithms.
        """,
        
        "qft": """
[bold cyan]📊 Quantum Fourier Transform (QFT)[/bold cyan]

[yellow]What it does:[/yellow]
Quantum analog of the discrete Fourier transform.
Transforms computational basis states to Fourier basis.

[yellow]Key concepts:[/yellow]
1. [green]Hadamard gates:[/green] Create superposition
2. [green]Controlled rotations:[/green] Apply phase shifts based on qubit values
3. [green]SWAP gates:[/green] Reverse qubit order at the end

[yellow]Circuit structure:[/yellow]
For each qubit i (from 0 to n-1):
• Apply H to qubit i
• Apply controlled R_k rotations from qubits j > i to i
• Final SWAP to reverse order

[yellow]Why it's important:[/yellow]
• Core subroutine in Shor's factoring algorithm
• Used in Quantum Phase Estimation
• Enables exponential speedup in period finding
        """,
        
        "vqe": """
[bold cyan]⚡ Variational Quantum Eigensolver (VQE)[/bold cyan]

[yellow]What it does:[/yellow]
Hybrid quantum-classical algorithm to find ground state energy of molecules.
Uses quantum computer for state preparation, classical computer for optimization.

[yellow]Key concepts:[/yellow]
1. [green]Ansatz:[/green] Parameterized quantum circuit (trial wavefunction)
2. [green]Cost function:[/green] ⟨ψ(θ)|H|ψ(θ)⟩ - expectation value of Hamiltonian
3. [green]Optimization:[/green] Classical optimizer adjusts parameters θ

[yellow]Algorithm flow:[/yellow]
1. Prepare parameterized state |ψ(θ)⟩ on quantum computer
2. Measure expectation value of Hamiltonian
3. Classical optimizer updates θ to minimize energy
4. Repeat until convergence

[yellow]Why it's important:[/yellow]
• Near-term quantum algorithm (works on NISQ devices)
• Applications in chemistry and materials science
• Foundation for quantum machine learning
        """,
        
        "bell": """
[bold cyan]🔔 Bell States[/bold cyan]

[yellow]What they are:[/yellow]
The four maximally entangled two-qubit states.
Named after physicist John Bell.

[yellow]The four Bell states:[/yellow]
• |Φ+⟩ = (|00⟩ + |11⟩)/√2
• |Φ-⟩ = (|00⟩ - |11⟩)/√2
• |Ψ+⟩ = (|01⟩ + |10⟩)/√2
• |Ψ-⟩ = (|01⟩ - |10⟩)/√2

[yellow]Creation circuit:[/yellow]
• Apply H to first qubit
• Apply CNOT (control: first, target: second)

[yellow]Why they're important:[/yellow]
• Demonstrate quantum entanglement
• Used in quantum teleportation
• Basis for Bell inequality tests
• Foundation for quantum cryptography
        """,
        
        "ghz": """
[bold cyan]🌟 GHZ State (Greenberger-Horne-Zeilinger)[/bold cyan]

[yellow]What it is:[/yellow]
Multi-qubit generalization of Bell states.
|GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2

[yellow]Properties:[/yellow]
• Maximally entangled n-qubit state
• If any qubit is measured, the state collapses
• "All or nothing" correlation

[yellow]Creation circuit:[/yellow]
• Apply H to first qubit
• Apply CNOT from qubit 0 to 1
• Apply CNOT from qubit 1 to 2
• ... and so on

[yellow]Why it's important:[/yellow]
• Tests foundations of quantum mechanics
• Used in quantum error correction
• Demonstrates genuine multipartite entanglement
• Applications in quantum metrology
        """
    }
    
    if algorithm.lower() not in explanations:
        console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
        console.print(f"[dim]Available: {', '.join(explanations.keys())}[/dim]")
        raise typer.Exit(1)
    
    console.print(explanations[algorithm.lower()])


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@animate_app.command("evolution")
def animate_evolution(
    algorithm: str = typer.Option("grover", "--algorithm", "-a", help="Algorithm to animate"),
    qubits: int = typer.Option(2, "--qubits", "-q", help="Number of qubits"),
    export_path: Optional[str] = typer.Option(None, "--export", "-e", help="Export animation (GIF/MP4)"),
    theme: str = typer.Option("quantum_dark", "--theme", help="Visualization theme"),
    terminal: bool = typer.Option(False, "--terminal", "-t", help="Terminal ASCII animation"),
):
    """
    🎬 Create animated visualizations of quantum state evolution.
    
    Examples:
    
        quantumviz animate evolution --algorithm grover --qubits 2
        
        quantumviz animate evolution -a qft -q 3 --export animation.gif
        
        quantumviz animate evolution --terminal
    """
    from quantumviz.core.algorithms import QuantumAlgorithms
    from quantumviz.core.simulator import QuantumSimulator
    from quantumviz.visualization.animation import QuantumAnimator
    from quantumviz.visualization.themes import VisualizationTheme
    
    console.print(Panel.fit(
        f"[bold cyan]🎬 Animation: {algorithm.upper()}[/bold cyan]\n\n"
        f"Qubits: [green]{qubits}[/green]",
        title="Animation Configuration"
    ))
    
    # Build and run circuit
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Building circuit...", total=None)
        
        if algorithm == "grover":
            circuit, _ = QuantumAlgorithms.grover_search(qubits, [1])
        elif algorithm == "qft":
            circuit = QuantumAlgorithms.qft(qubits)
        elif algorithm == "bell":
            circuit = QuantumAlgorithms.bell_state_circuit()
            qubits = 2
        elif algorithm == "ghz":
            circuit = QuantumAlgorithms.ghz_state_circuit(qubits)
        else:
            console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
            raise typer.Exit(1)
        
        progress.update(task, description="[cyan]Running simulation...")
        
        simulator = QuantumSimulator()
        simulator.enable_history_capture(True)
        result = simulator.run(circuit, shots=1000)
        
        progress.update(task, description="[green]✓ Simulation complete!")
    
    if not result.state_history:
        console.print("[yellow]Warning: No state history captured[/yellow]")
        return
    
    console.print(f"\n[bold]States captured:[/bold] {len(result.state_history)}")
    
    viz_theme = VisualizationTheme.get_theme(theme)
    animator = QuantumAnimator(theme=viz_theme)
    
    if terminal:
        console.print("\n[cyan]Starting terminal animation (Ctrl+C to stop)...[/cyan]\n")
        animator.terminal_animation(result.state_history, delay=0.5)
    else:
        # Use matplotlib animation
        animator.animate_bloch_sphere(
            result.state_history,
            output_path=export_path,
            show=export_path is None
        )
        
        if export_path:
            console.print(f"[green]✓ Animation saved to {export_path}[/green]")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@export_app.command("state")
def export_state(
    state_type: str = typer.Option("random", "--state", "-s", help="State type"),
    qubits: int = typer.Option(2, "--qubits", "-q", help="Number of qubits"),
    output: str = typer.Option("state.json", "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, csv"),
):
    """
    📤 Export quantum state data to file.
    
    Examples:
    
        quantumviz export state --state bell_phi+ --output bell.json
        
        quantumviz export state -s random -q 3 -o state.csv -f csv
    """
    from quantumviz.core.quantum_state import QuantumState
    import json
    
    # Create state
    if state_type == "random":
        state = QuantumState.random_state(qubits)
    elif state_type.startswith("bell_"):
        state = QuantumState.bell_state(state_type.replace("bell_", ""))
    elif state_type.startswith("ghz"):
        n = int(state_type.replace("ghz", "") or str(qubits))
        state = QuantumState.ghz_state(n)
    else:
        state = QuantumState.from_basis("0" * qubits)
    
    if format == "json":
        with open(output, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
    elif format == "csv":
        import csv
        probs = state.get_probabilities()
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["state", "probability"])
            for i, p in enumerate(probs):
                writer.writerow([format(i, f'0{state.num_qubits}b'), p])
    
    console.print(f"[green]✓ Exported to {output}[/green]")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _display_algorithm_results(result, qubits: int):
    """Display algorithm results in a formatted table."""
    console.print(f"\n[bold]Algorithm:[/bold] {result.name}")
    console.print(f"[bold]Iterations:[/bold] {result.iterations}")
    console.print(f"[bold]Success Probability:[/bold] [green]{result.success_probability:.2%}[/green]")
    
    if result.classical_result is not None:
        console.print(f"[bold]Found State:[/bold] [yellow]{result.classical_result}[/yellow] "
                     f"(binary: {format(result.classical_result, f'0{qubits}b')})")
    
    if result.simulation_result.counts:
        console.print("\n[bold]Measurement Results:[/bold]")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("State", style="yellow")
        table.add_column("Counts", style="green")
        table.add_column("Probability", style="magenta")
        table.add_column("Bar", style="cyan")
        
        total = sum(result.simulation_result.counts.values())
        for state, count in sorted(result.simulation_result.counts.items(), key=lambda x: -x[1])[:10]:
            prob = count / total
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            table.add_row(f"|{state}⟩", str(count), f"{prob:.2%}", bar)
        
        console.print(table)


if __name__ == "__main__":
    app()
