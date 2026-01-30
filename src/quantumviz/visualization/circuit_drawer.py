"""
Circuit Drawer
==============

Beautiful quantum circuit visualizations with multiple styles.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from quantumviz.visualization.themes import VisualizationTheme


@dataclass
class GateVisual:
    """Visual representation of a quantum gate."""
    name: str
    width: float = 1.0
    height: float = 0.6
    color: str = "#4A90D9"
    text_color: str = "#FFFFFF"
    style: str = "box"  # "box", "circle", "diamond"


class CircuitDrawer:
    """
    Renders quantum circuits as beautiful diagrams.
    
    Supports:
    - Matplotlib rendering
    - ASCII art for terminal
    - SVG export
    - LaTeX/Qcircuit format
    """
    
    GATE_VISUALS = {
        "H": GateVisual("H", color="#4A90D9"),
        "X": GateVisual("X", color="#D94A4A"),
        "Y": GateVisual("Y", color="#4AD94A"),
        "Z": GateVisual("Z", color="#D9D94A"),
        "S": GateVisual("S", color="#9B59B6"),
        "T": GateVisual("T", color="#E67E22"),
        "RX": GateVisual("Rx", color="#4A90D9"),
        "RY": GateVisual("Ry", color="#4AD94A"),
        "RZ": GateVisual("Rz", color="#D9D94A"),
        "CNOT": GateVisual("⊕", style="circle"),
        "CZ": GateVisual("Z", color="#D9D94A"),
        "SWAP": GateVisual("×", style="circle"),
        "Toffoli": GateVisual("⊕", style="circle"),
        "MEASURE": GateVisual("M", color="#95A5A6"),
    }
    
    def __init__(self, theme: Optional[VisualizationTheme] = None):
        """Initialize drawer with theme."""
        self.theme = theme or VisualizationTheme.quantum_dark()
    
    def draw(self, circuit, output_path: Optional[str] = None, 
             show: bool = True, figsize: Optional[Tuple[float, float]] = None) -> Any:
        """
        Draw quantum circuit using matplotlib.
        
        Args:
            circuit: QuantumCircuit to draw
            output_path: Optional path to save figure
            show: Whether to display
            figsize: Figure size (auto-calculated if None)
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, Circle
        
        n_qubits = circuit.num_qubits
        depth = circuit.depth
        
        if figsize is None:
            figsize = (max(8, depth * 1.5), max(4, n_qubits * 1.2))
        
        plt.style.use('dark_background' if 'dark' in self.theme.name.value else 'default')
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.theme.colors.background)
        ax.set_facecolor(self.theme.colors.background)
        
        wire_spacing = self.theme.wire_spacing
        gate_width = self.theme.gate_width
        gate_height = self.theme.gate_height
        
        # Draw qubit wires
        for i in range(n_qubits):
            y = (n_qubits - 1 - i) * wire_spacing
            ax.hlines(y, -0.5, depth + 0.5, 
                     color=self.theme.colors.grid, linewidth=1.5, zorder=1)
            ax.text(-1, y, f'q{i}', ha='right', va='center',
                   color=self.theme.colors.text, fontsize=self.theme.label_size)
        
        # Track which positions are occupied
        qubit_positions = {i: 0 for i in range(n_qubits)}
        
        # Draw gates
        for layer_idx, layer in enumerate(circuit.layers):
            x = layer_idx + 0.5
            
            for instr in layer.instructions:
                targets = instr.target_qubits
                gate_name = instr.gate_name
                
                if gate_name == "BARRIER":
                    for q in targets:
                        y = (n_qubits - 1 - q) * wire_spacing
                        ax.axvline(x, y - 0.4, y + 0.4, 
                                  color=self.theme.colors.secondary, 
                                  linestyle='--', alpha=0.5, linewidth=1)
                    continue
                
                if gate_name == "MEASURE":
                    q = targets[0]
                    y = (n_qubits - 1 - q) * wire_spacing
                    self._draw_measure(ax, x, y)
                    continue
                
                visual = self.GATE_VISUALS.get(gate_name, GateVisual(gate_name[:3]))
                
                if len(targets) == 1:
                    # Single-qubit gate
                    q = targets[0]
                    y = (n_qubits - 1 - q) * wire_spacing
                    self._draw_single_gate(ax, x, y, visual, instr.params)
                
                elif len(targets) == 2:
                    # Two-qubit gate
                    q1, q2 = targets
                    y1 = (n_qubits - 1 - q1) * wire_spacing
                    y2 = (n_qubits - 1 - q2) * wire_spacing
                    
                    if gate_name in ["CNOT", "CX"]:
                        self._draw_cnot(ax, x, y1, y2)
                    elif gate_name == "CZ":
                        self._draw_cz(ax, x, y1, y2)
                    elif gate_name == "SWAP":
                        self._draw_swap(ax, x, y1, y2)
                    else:
                        self._draw_controlled_gate(ax, x, y1, y2, visual, instr.params)
                
                elif len(targets) == 3:
                    # Three-qubit gate (Toffoli, Fredkin)
                    ys = [(n_qubits - 1 - q) * wire_spacing for q in targets]
                    if gate_name in ["Toffoli", "CCNOT"]:
                        self._draw_toffoli(ax, x, ys)
                    else:
                        self._draw_multi_gate(ax, x, ys, visual)
        
        # Set limits and styling
        ax.set_xlim(-1.5, depth + 1)
        ax.set_ylim(-0.5, n_qubits * wire_spacing)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        ax.set_title(circuit.name, color=self.theme.colors.text,
                    fontsize=self.theme.title_size, pad=20)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150, 
                       facecolor=self.theme.colors.background,
                       bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def _draw_single_gate(self, ax, x: float, y: float, 
                          visual: GateVisual, params: Optional[Dict] = None):
        """Draw a single-qubit gate."""
        from matplotlib.patches import FancyBboxPatch
        
        w, h = self.theme.gate_width * 0.8, self.theme.gate_height
        
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=visual.color,
            edgecolor=self.theme.colors.text,
            linewidth=1.5,
            zorder=10
        )
        ax.add_patch(rect)
        
        # Gate label
        label = visual.name
        if params:
            if 'theta' in params:
                label += f"\n{params['theta']:.2f}"
        
        ax.text(x, y, label, ha='center', va='center',
               color=visual.text_color, fontsize=self.theme.tick_size,
               fontweight='bold', zorder=11)
    
    def _draw_cnot(self, ax, x: float, y_ctrl: float, y_targ: float):
        """Draw CNOT gate."""
        from matplotlib.patches import Circle
        
        # Control dot
        ctrl = Circle((x, y_ctrl), 0.1, color=self.theme.colors.primary, zorder=10)
        ax.add_patch(ctrl)
        
        # Target (XOR symbol)
        radius = 0.2
        target_circle = Circle((x, y_targ), radius, fill=False,
                               color=self.theme.colors.primary, linewidth=2, zorder=10)
        ax.add_patch(target_circle)
        
        # Plus inside target
        ax.plot([x - radius, x + radius], [y_targ, y_targ],
               color=self.theme.colors.primary, linewidth=2, zorder=11)
        ax.plot([x, x], [y_targ - radius, y_targ + radius],
               color=self.theme.colors.primary, linewidth=2, zorder=11)
        
        # Connecting line
        ax.vlines(x, min(y_ctrl, y_targ) + 0.1, max(y_ctrl, y_targ) - 0.1,
                 color=self.theme.colors.primary, linewidth=2, zorder=5)
    
    def _draw_cz(self, ax, x: float, y1: float, y2: float):
        """Draw CZ gate."""
        from matplotlib.patches import Circle
        
        # Both are control dots for CZ
        for y in [y1, y2]:
            ctrl = Circle((x, y), 0.1, color=self.theme.colors.tertiary, zorder=10)
            ax.add_patch(ctrl)
        
        # Connecting line
        ax.vlines(x, min(y1, y2) + 0.1, max(y1, y2) - 0.1,
                 color=self.theme.colors.tertiary, linewidth=2, zorder=5)
    
    def _draw_swap(self, ax, x: float, y1: float, y2: float):
        """Draw SWAP gate."""
        size = 0.15
        
        for y in [y1, y2]:
            ax.plot([x - size, x + size], [y - size, y + size],
                   color=self.theme.colors.secondary, linewidth=2, zorder=10)
            ax.plot([x - size, x + size], [y + size, y - size],
                   color=self.theme.colors.secondary, linewidth=2, zorder=10)
        
        ax.vlines(x, min(y1, y2), max(y1, y2),
                 color=self.theme.colors.secondary, linewidth=2, zorder=5)
    
    def _draw_controlled_gate(self, ax, x: float, y_ctrl: float, y_targ: float,
                              visual: GateVisual, params: Optional[Dict] = None):
        """Draw generic controlled gate."""
        from matplotlib.patches import Circle
        
        # Control dot
        ctrl = Circle((x, y_ctrl), 0.1, color=visual.color, zorder=10)
        ax.add_patch(ctrl)
        
        # Target gate
        self._draw_single_gate(ax, x, y_targ, visual, params)
        
        # Connecting line
        ax.vlines(x, min(y_ctrl, y_targ) + 0.1, max(y_ctrl, y_targ) - 0.3,
                 color=visual.color, linewidth=2, zorder=5)
    
    def _draw_toffoli(self, ax, x: float, ys: List[float]):
        """Draw Toffoli (CCNOT) gate."""
        from matplotlib.patches import Circle
        
        y_ctrl1, y_ctrl2, y_targ = ys
        
        # Control dots
        for y in [y_ctrl1, y_ctrl2]:
            ctrl = Circle((x, y), 0.1, color=self.theme.colors.primary, zorder=10)
            ax.add_patch(ctrl)
        
        # Target (XOR symbol)
        radius = 0.2
        target_circle = Circle((x, y_targ), radius, fill=False,
                               color=self.theme.colors.primary, linewidth=2, zorder=10)
        ax.add_patch(target_circle)
        ax.plot([x - radius, x + radius], [y_targ, y_targ],
               color=self.theme.colors.primary, linewidth=2, zorder=11)
        ax.plot([x, x], [y_targ - radius, y_targ + radius],
               color=self.theme.colors.primary, linewidth=2, zorder=11)
        
        # Connecting line
        ax.vlines(x, min(ys), max(ys),
                 color=self.theme.colors.primary, linewidth=2, zorder=5)
    
    def _draw_multi_gate(self, ax, x: float, ys: List[float], visual: GateVisual):
        """Draw multi-qubit gate as a box spanning qubits."""
        from matplotlib.patches import FancyBboxPatch
        
        y_min, y_max = min(ys), max(ys)
        w = self.theme.gate_width * 0.8
        h = y_max - y_min + self.theme.gate_height
        
        rect = FancyBboxPatch(
            (x - w/2, y_min - self.theme.gate_height/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=visual.color,
            edgecolor=self.theme.colors.text,
            linewidth=1.5,
            zorder=10
        )
        ax.add_patch(rect)
        
        ax.text(x, (y_min + y_max) / 2, visual.name,
               ha='center', va='center', color=visual.text_color,
               fontsize=self.theme.tick_size, fontweight='bold', zorder=11)
    
    def _draw_measure(self, ax, x: float, y: float):
        """Draw measurement symbol."""
        from matplotlib.patches import FancyBboxPatch, Arc
        
        w, h = self.theme.gate_width * 0.8, self.theme.gate_height
        
        # Box
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02",
            facecolor=self.theme.colors.surface,
            edgecolor=self.theme.colors.text,
            linewidth=1.5,
            zorder=10
        )
        ax.add_patch(rect)
        
        # Meter arc
        arc = Arc((x, y - 0.05), 0.3, 0.25, angle=0,
                 theta1=0, theta2=180,
                 color=self.theme.colors.text, linewidth=1.5, zorder=11)
        ax.add_patch(arc)
        
        # Meter needle
        ax.plot([x, x + 0.12], [y - 0.05, y + 0.15],
               color=self.theme.colors.text, linewidth=1.5, zorder=11)
    
    def to_ascii(self, circuit) -> str:
        """
        Convert circuit to ASCII art.
        
        Args:
            circuit: QuantumCircuit to convert
            
        Returns:
            ASCII representation string
        """
        n = circuit.num_qubits
        
        lines = [f"q{i}: ─" for i in range(n)]
        
        for layer in circuit.layers:
            # Find max width for this layer
            max_width = 1
            
            for instr in layer.instructions:
                gate = instr.gate_name
                if gate == "BARRIER":
                    for q in instr.target_qubits:
                        lines[q] += "│"
                elif gate == "MEASURE":
                    q = instr.target_qubits[0]
                    lines[q] += "┤M├"
                elif len(instr.target_qubits) == 1:
                    q = instr.target_qubits[0]
                    name = gate[:3].center(3)
                    lines[q] += f"┤{name}├"
                elif gate in ["CNOT", "CX"]:
                    q1, q2 = instr.target_qubits
                    lines[q1] += "──●──"
                    lines[q2] += "──⊕──"
                elif gate == "CZ":
                    q1, q2 = instr.target_qubits
                    lines[q1] += "──●──"
                    lines[q2] += "──●──"
                elif gate == "SWAP":
                    q1, q2 = instr.target_qubits
                    lines[q1] += "──×──"
                    lines[q2] += "──×──"
                else:
                    for q in instr.target_qubits:
                        lines[q] += f"─[{gate[:2]}]─"
            
            # Pad all lines to same length
            max_len = max(len(line) for line in lines)
            lines = [line.ljust(max_len, '─') for line in lines]
        
        # Add final wire
        lines = [line + "─" for line in lines]
        
        # Build header
        header = f"╔{'═' * (max(len(l) for l in lines) + 2)}╗"
        title = f"║ {circuit.name.center(max(len(l) for l in lines))} ║"
        sep = f"╠{'═' * (max(len(l) for l in lines) + 2)}╣"
        footer = f"╚{'═' * (max(len(l) for l in lines) + 2)}╝"
        
        result = [header, title, sep]
        for line in lines:
            result.append(f"║ {line} ║")
        result.append(footer)
        
        return '\n'.join(result)
    
    def to_latex(self, circuit) -> str:
        """
        Convert circuit to LaTeX Qcircuit format.
        
        Args:
            circuit: QuantumCircuit to convert
            
        Returns:
            LaTeX string
        """
        n = circuit.num_qubits
        
        lines = ["\\Qcircuit @C=1em @R=.7em {"]
        
        for i in range(n):
            row = f"  & \\lstick{{q_{i}}} "
            
            for layer in circuit.layers:
                found = False
                for instr in layer.instructions:
                    if i in instr.target_qubits:
                        gate = instr.gate_name
                        
                        if gate == "H":
                            row += "& \\gate{H} "
                        elif gate == "X":
                            row += "& \\gate{X} "
                        elif gate == "Y":
                            row += "& \\gate{Y} "
                        elif gate == "Z":
                            row += "& \\gate{Z} "
                        elif gate == "MEASURE":
                            row += "& \\meter "
                        elif gate in ["CNOT", "CX"]:
                            q1, q2 = instr.target_qubits
                            if i == q1:
                                diff = q2 - q1
                                row += f"& \\ctrl{{{diff}}} "
                            else:
                                row += "& \\targ "
                        else:
                            row += f"& \\gate{{{gate}}} "
                        
                        found = True
                        break
                
                if not found:
                    row += "& \\qw "
            
            row += "& \\qw \\\\"
            lines.append(row)
        
        lines.append("}")
        
        return '\n'.join(lines)
