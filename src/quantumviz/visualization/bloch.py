"""
Bloch Sphere Renderer
=====================

Stunning 3D Bloch sphere visualizations for quantum states.
Features:
- Interactive 3D rendering
- Multiple qubit support
- State trajectory visualization
- Customizable themes
- Export to PNG, SVG, HTML
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import io

from quantumviz.core.quantum_state import QuantumState, BlochCoordinates
from quantumviz.visualization.themes import VisualizationTheme


@dataclass
class BlochVector:
    """Represents a vector on the Bloch sphere."""
    coords: BlochCoordinates
    color: str = "#00FF88"
    label: str = ""
    alpha: float = 1.0
    show_projection: bool = True


class BlochSphereRenderer:
    """
    High-quality 3D Bloch sphere renderer.
    
    Renders quantum states as vectors on the Bloch sphere with
    support for animations, trajectories, and multiple qubits.
    """
    
    def __init__(self, theme: Optional[VisualizationTheme] = None):
        """
        Initialize renderer.
        
        Args:
            theme: Visualization theme (default: quantum_dark)
        """
        self.theme = theme or VisualizationTheme.quantum_dark()
        self._fig = None
        self._ax = None
    
    def _setup_figure(self, figsize: Tuple[float, float] = (10, 10)):
        """Setup matplotlib figure and 3D axes."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        plt.style.use('dark_background' if 'dark' in self.theme.name.value else 'default')
        
        self._fig = plt.figure(figsize=figsize, facecolor=self.theme.colors.background)
        self._ax = self._fig.add_subplot(111, projection='3d', facecolor=self.theme.colors.background)
        
        # Set viewing angle
        self._ax.view_init(elev=self.theme.elevation, azim=self.theme.azimuth)
        
        # Remove axis labels for cleaner look
        self._ax.set_axis_off()
        
        return self._fig, self._ax
    
    def _draw_sphere(self, ax, alpha: float = None):
        """Draw the Bloch sphere surface."""
        if alpha is None:
            alpha = self.theme.sphere_alpha
        
        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, self.theme.sphere_resolution)
        v = np.linspace(0, np.pi, self.theme.sphere_resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Draw sphere surface
        ax.plot_surface(
            x, y, z,
            color=self.theme.colors.bloch_sphere,
            alpha=alpha,
            linewidth=0,
            antialiased=True
        )
        
        # Draw wireframe
        ax.plot_wireframe(
            x, y, z,
            color=self.theme.colors.bloch_wireframe,
            alpha=self.theme.wireframe_alpha,
            linewidth=0.5,
            rstride=5,
            cstride=5
        )
    
    def _draw_axes(self, ax):
        """Draw the X, Y, Z axes."""
        axis_length = 1.3
        
        # Draw axes
        colors = [self.theme.colors.state_0, self.theme.colors.state_1, self.theme.colors.superposition]
        labels = ['X', 'Y', 'Z']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            direction = [0, 0, 0]
            direction[i] = axis_length
            
            # Positive direction
            ax.quiver(0, 0, 0, *direction, 
                     color=color, alpha=0.6, arrow_length_ratio=0.1, linewidth=1.5)
            
            # Negative direction
            neg_direction = [-d for d in direction]
            ax.quiver(0, 0, 0, *neg_direction,
                     color=color, alpha=0.3, arrow_length_ratio=0.1, linewidth=1)
            
            # Labels
            label_pos = [d * 1.1 for d in direction]
            ax.text(*label_pos, f'|{label.lower()}+⟩', color=color, fontsize=self.theme.label_size)
            
            neg_label_pos = [d * 1.1 for d in neg_direction]
            ax.text(*neg_label_pos, f'|{label.lower()}-⟩', color=color, fontsize=self.theme.label_size, alpha=0.6)
        
        # Special state labels
        ax.text(0, 0, 1.2, '|0⟩', color=self.theme.colors.text, fontsize=self.theme.label_size, ha='center')
        ax.text(0, 0, -1.2, '|1⟩', color=self.theme.colors.text, fontsize=self.theme.label_size, ha='center')
    
    def _draw_equator_and_meridians(self, ax):
        """Draw reference circles on the sphere."""
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Equator (XY plane)
        ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                color=self.theme.colors.grid, alpha=0.4, linewidth=1)
        
        # XZ meridian
        ax.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta),
                color=self.theme.colors.grid, alpha=0.3, linewidth=0.5)
        
        # YZ meridian
        ax.plot(np.zeros_like(theta), np.cos(theta), np.sin(theta),
                color=self.theme.colors.grid, alpha=0.3, linewidth=0.5)
    
    def _draw_state_vector(self, ax, coords: BlochCoordinates, 
                           color: str = None, label: str = "", 
                           alpha: float = 1.0, show_projection: bool = True):
        """Draw a state vector on the Bloch sphere."""
        if color is None:
            color = self.theme.colors.bloch_vector
        
        x, y, z = coords.x, coords.y, coords.z
        
        # Draw main vector with arrow
        ax.quiver(0, 0, 0, x, y, z,
                 color=color, alpha=alpha,
                 arrow_length_ratio=0.15,
                 linewidth=self.theme.vector_width)
        
        # Draw point at tip
        ax.scatter([x], [y], [z], color=color, s=self.theme.point_size, alpha=alpha)
        
        # Draw projection lines if enabled
        if show_projection:
            # Projection onto XY plane
            ax.plot([x, x], [y, y], [0, z], 
                   color=color, alpha=0.3, linestyle='--', linewidth=1)
            ax.plot([0, x], [0, y], [0, 0],
                   color=color, alpha=0.3, linestyle='--', linewidth=1)
            
            # Small marker on XY plane
            ax.scatter([x], [y], [0], color=color, s=30, alpha=0.3, marker='o')
        
        # Label if provided
        if label:
            ax.text(x * 1.15, y * 1.15, z * 1.15, label,
                   color=color, fontsize=self.theme.label_size)
        
        # Add glow effect if enabled
        if self.theme.glow_effect:
            for i in range(3):
                scale = 1 + i * 0.1
                ax.scatter([x], [y], [z], color=color, 
                          s=self.theme.point_size * scale, 
                          alpha=0.1 / (i + 1))
    
    def _draw_trajectory(self, ax, trajectory: List[BlochCoordinates], 
                        color: str = None, alpha: float = 0.5):
        """Draw a state trajectory on the sphere."""
        if color is None:
            color = self.theme.colors.bloch_vector
        
        if len(trajectory) < 2:
            return
        
        xs = [c.x for c in trajectory]
        ys = [c.y for c in trajectory]
        zs = [c.z for c in trajectory]
        
        # Draw line with fading alpha
        for i in range(len(trajectory) - 1):
            segment_alpha = alpha * (i + 1) / len(trajectory)
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [zs[i], zs[i+1]],
                   color=color, alpha=segment_alpha, linewidth=2)
        
        # Mark start and end points
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=color, s=50, alpha=0.5, marker='o')
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=color, s=100, alpha=1.0, marker='*')
    
    def render_state(self, 
                     state: QuantumState,
                     qubit_index: int = 0,
                     title: str = None,
                     show_axes: bool = True,
                     show_labels: bool = True,
                     figsize: Tuple[float, float] = (10, 10)) -> Any:
        """
        Render a single quantum state on the Bloch sphere.
        
        Args:
            state: Quantum state to visualize
            qubit_index: Which qubit to show (for multi-qubit states)
            title: Plot title
            show_axes: Whether to show X, Y, Z axes
            show_labels: Whether to show state labels
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = self._setup_figure(figsize)
        
        # Draw sphere elements
        self._draw_sphere(ax)
        self._draw_equator_and_meridians(ax)
        
        if show_axes:
            self._draw_axes(ax)
        
        # Get Bloch coordinates
        coords = state.get_bloch_coordinates(qubit_index)
        
        # Draw state vector
        label = state.name if show_labels else ""
        self._draw_state_vector(ax, coords, label=label)
        
        # Add title
        if title:
            ax.set_title(title, color=self.theme.colors.text, 
                        fontsize=self.theme.title_size, pad=20)
        
        # Add state info
        info_text = f"θ = {coords.theta:.3f}\nφ = {coords.phi:.3f}"
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 color=self.theme.colors.text_secondary,
                 fontsize=self.theme.tick_size,
                 verticalalignment='top',
                 fontfamily=self.theme.font_family)
        
        # Set axis limits
        limit = 1.5
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def render_multiple_states(self,
                               states: List[QuantumState],
                               labels: Optional[List[str]] = None,
                               colors: Optional[List[str]] = None,
                               title: str = None,
                               figsize: Tuple[float, float] = (12, 10)) -> Any:
        """
        Render multiple quantum states on the same Bloch sphere.
        
        Args:
            states: List of quantum states
            labels: Optional labels for each state
            colors: Optional colors for each state
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = self._setup_figure(figsize)
        
        # Draw sphere elements
        self._draw_sphere(ax)
        self._draw_equator_and_meridians(ax)
        self._draw_axes(ax)
        
        # Default colors if not provided
        if colors is None:
            default_colors = [
                self.theme.colors.bloch_vector,
                self.theme.colors.secondary,
                self.theme.colors.tertiary,
                self.theme.colors.state_0,
                self.theme.colors.state_1,
            ]
            colors = [default_colors[i % len(default_colors)] for i in range(len(states))]
        
        if labels is None:
            labels = [s.name for s in states]
        
        # Draw each state
        for state, label, color in zip(states, labels, colors):
            coords = state.get_bloch_coordinates(0)
            self._draw_state_vector(ax, coords, color=color, label=label)
        
        if title:
            ax.set_title(title, color=self.theme.colors.text,
                        fontsize=self.theme.title_size, pad=20)
        
        # Set limits
        limit = 1.5
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def render_trajectory(self,
                         states: List[QuantumState],
                         qubit_index: int = 0,
                         title: str = "Quantum State Evolution",
                         show_intermediate: bool = True,
                         figsize: Tuple[float, float] = (12, 10)) -> Any:
        """
        Render a trajectory of quantum states showing evolution.
        
        Args:
            states: List of states in temporal order
            qubit_index: Which qubit to visualize
            title: Plot title
            show_intermediate: Show intermediate points
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = self._setup_figure(figsize)
        
        # Draw sphere
        self._draw_sphere(ax)
        self._draw_equator_and_meridians(ax)
        self._draw_axes(ax)
        
        # Get trajectory coordinates
        trajectory = [s.get_bloch_coordinates(qubit_index) for s in states]
        
        # Draw trajectory
        self._draw_trajectory(ax, trajectory)
        
        # Draw intermediate points if requested
        if show_intermediate and len(trajectory) > 2:
            for i, coords in enumerate(trajectory[1:-1], 1):
                alpha = 0.3 + 0.5 * (i / len(trajectory))
                ax.scatter([coords.x], [coords.y], [coords.z],
                          color=self.theme.colors.bloch_vector,
                          s=30, alpha=alpha)
        
        # Draw initial and final states prominently
        if trajectory:
            self._draw_state_vector(ax, trajectory[0], 
                                   color=self.theme.colors.state_0,
                                   label="Initial", alpha=0.7)
            self._draw_state_vector(ax, trajectory[-1],
                                   color=self.theme.colors.state_1,
                                   label="Final")
        
        if title:
            ax.set_title(title, color=self.theme.colors.text,
                        fontsize=self.theme.title_size, pad=20)
        
        # Add evolution info
        info_text = f"Steps: {len(states)}"
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 color=self.theme.colors.text_secondary,
                 fontsize=self.theme.tick_size,
                 verticalalignment='top')
        
        # Set limits
        limit = 1.5
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def render_multi_qubit(self,
                          state: QuantumState,
                          title: str = None,
                          figsize: Tuple[float, float] = None) -> Any:
        """
        Render Bloch spheres for all qubits in a multi-qubit state.
        
        Args:
            state: Multi-qubit quantum state
            title: Plot title
            figsize: Figure size (auto-calculated if None)
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        n = state.num_qubits
        
        # Calculate figure layout
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        
        if figsize is None:
            figsize = (5 * cols, 5 * rows)
        
        plt.style.use('dark_background' if 'dark' in self.theme.name.value else 'default')
        fig = plt.figure(figsize=figsize, facecolor=self.theme.colors.background)
        
        for i in range(n):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d',
                               facecolor=self.theme.colors.background)
            ax.view_init(elev=self.theme.elevation, azim=self.theme.azimuth)
            ax.set_axis_off()
            
            # Draw sphere
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x, y, z, color=self.theme.colors.bloch_sphere,
                          alpha=self.theme.sphere_alpha, linewidth=0)
            ax.plot_wireframe(x, y, z, color=self.theme.colors.bloch_wireframe,
                            alpha=0.2, linewidth=0.3, rstride=5, cstride=5)
            
            # Draw state vector
            coords = state.get_bloch_coordinates(i)
            ax.quiver(0, 0, 0, coords.x, coords.y, coords.z,
                     color=self.theme.colors.bloch_vector,
                     arrow_length_ratio=0.15, linewidth=2)
            ax.scatter([coords.x], [coords.y], [coords.z],
                      color=self.theme.colors.bloch_vector, s=50)
            
            ax.set_title(f"Qubit {i}", color=self.theme.colors.text,
                        fontsize=self.theme.label_size)
            
            limit = 1.3
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])
            ax.set_box_aspect([1, 1, 1])
        
        if title:
            fig.suptitle(title, color=self.theme.colors.text,
                        fontsize=self.theme.title_size)
        
        plt.tight_layout()
        return fig
    
    def save(self, fig: Any, path: str, dpi: int = 150, transparent: bool = False):
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            path: Output path (supports .png, .svg, .pdf)
            dpi: Resolution for raster formats
            transparent: Transparent background
        """
        fig.savefig(
            path,
            dpi=dpi,
            facecolor=self.theme.colors.background if not transparent else 'none',
            edgecolor='none',
            bbox_inches='tight',
            transparent=transparent
        )
    
    def to_ascii(self, state: QuantumState, qubit_index: int = 0) -> str:
        """
        Generate ASCII art representation of Bloch sphere.
        
        Args:
            state: Quantum state
            qubit_index: Which qubit to show
            
        Returns:
            ASCII art string
        """
        coords = state.get_bloch_coordinates(qubit_index)
        
        # Simplified ASCII Bloch sphere
        lines = [
            "          |0⟩",
            "           ↑",
            "         _____",
            "       /       \\",
            "      /    │    \\",
            "     |     │     |  ← |y+⟩",
            "─────●─────┼─────●───→ |x+⟩",
            "     |     │     |",
            "      \\    │    /",
            "       \\___|___/",
            "           ↓",
            "          |1⟩",
            "",
            f"State: {state.name}",
            f"θ = {coords.theta:.3f} rad ({np.degrees(coords.theta):.1f}°)",
            f"φ = {coords.phi:.3f} rad ({np.degrees(coords.phi):.1f}°)",
            f"(x, y, z) = ({coords.x:.3f}, {coords.y:.3f}, {coords.z:.3f})",
        ]
        
        return "\n".join(lines)
    
    def show(self, fig: Any = None):
        """Display the figure interactively."""
        import matplotlib.pyplot as plt
        
        if fig is None:
            fig = self._fig
        
        plt.show()
