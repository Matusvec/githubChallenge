"""
Quantum Animation Engine
========================

Creates stunning animated visualizations of quantum state evolution.
Features:
- Smooth Bloch sphere animations
- Probability distribution evolution
- Circuit execution step-by-step
- GIF and video export
- Real-time terminal animations
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import io
import time as time_module

from quantumviz.core.quantum_state import QuantumState, BlochCoordinates
from quantumviz.core.simulator import SimulationResult
from quantumviz.visualization.themes import VisualizationTheme


@dataclass
class AnimationFrame:
    """Single frame in an animation."""
    state: QuantumState
    time: float
    label: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class AnimationConfig:
    """Configuration for animation rendering."""
    fps: int = 30
    duration: float = 5.0  # seconds
    resolution: Tuple[int, int] = (800, 800)
    loop: bool = True
    trail_frames: int = 10
    interpolation: str = "linear"  # "linear", "smooth", "bounce"
    easing: str = "ease_in_out"


class QuantumAnimator:
    """
    Creates animated visualizations of quantum systems.
    
    Supports:
    - Bloch sphere state evolution
    - Probability histogram animations
    - Gate-by-gate circuit execution
    - Time-dependent Hamiltonian evolution
    - Custom transition effects
    """
    
    def __init__(self, theme: Optional[VisualizationTheme] = None):
        """
        Initialize animator.
        
        Args:
            theme: Visualization theme
        """
        self.theme = theme or VisualizationTheme.quantum_dark()
        self.config = AnimationConfig(fps=self.theme.fps)
    
    def _interpolate_bloch(self, 
                          start: BlochCoordinates, 
                          end: BlochCoordinates,
                          t: float,
                          method: str = "geodesic") -> BlochCoordinates:
        """
        Interpolate between two Bloch coordinates.
        
        Args:
            start: Starting coordinates
            end: Ending coordinates
            t: Interpolation parameter [0, 1]
            method: "linear", "geodesic", or "smooth"
            
        Returns:
            Interpolated coordinates
        """
        if method == "linear":
            # Simple linear interpolation
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            z = start.z + t * (end.z - start.z)
        
        elif method == "geodesic":
            # Geodesic interpolation on the sphere (SLERP)
            v1 = np.array([start.x, start.y, start.z])
            v2 = np.array([end.x, end.y, end.z])
            
            # Normalize (they should already be on the sphere)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            
            if n1 < 1e-10 or n2 < 1e-10:
                return BlochCoordinates(
                    x=start.x + t * (end.x - start.x),
                    y=start.y + t * (end.y - start.y),
                    z=start.z + t * (end.z - start.z)
                )
            
            v1 = v1 / n1
            v2 = v2 / n2
            
            # Compute angle
            dot = np.clip(np.dot(v1, v2), -1, 1)
            omega = np.arccos(dot)
            
            if omega < 1e-10:
                x, y, z = v1
            else:
                # SLERP formula
                s1 = np.sin((1 - t) * omega) / np.sin(omega)
                s2 = np.sin(t * omega) / np.sin(omega)
                v = s1 * v1 + s2 * v2
                
                # Interpolate magnitude
                r = n1 + t * (n2 - n1)
                v = v * r
                x, y, z = v
        
        elif method == "smooth":
            # Smooth easing
            t = t * t * (3 - 2 * t)  # Smoothstep
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            z = start.z + t * (end.z - start.z)
        
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        # Compute spherical angles
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            theta, phi = 0, 0
        else:
            theta = np.arccos(np.clip(z / r, -1, 1))
            phi = np.arctan2(y, x)
        
        return BlochCoordinates(
            x=float(x), y=float(y), z=float(z),
            theta=float(theta), phi=float(phi)
        )
    
    def _easing(self, t: float, method: str = "ease_in_out") -> float:
        """Apply easing function to time parameter."""
        if method == "linear":
            return t
        elif method == "ease_in":
            return t * t
        elif method == "ease_out":
            return 1 - (1 - t) ** 2
        elif method == "ease_in_out":
            return 3 * t**2 - 2 * t**3
        elif method == "bounce":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) ** 2
        else:
            return t
    
    def create_evolution_frames(self,
                               states: List[QuantumState],
                               qubit_index: int = 0,
                               frames_between: int = 10) -> List[AnimationFrame]:
        """
        Create interpolated frames between quantum states.
        
        Args:
            states: List of quantum states to animate between
            qubit_index: Which qubit to animate
            frames_between: Number of interpolation frames
            
        Returns:
            List of animation frames
        """
        frames = []
        total_time = 0.0
        
        for i, state in enumerate(states):
            # Add the actual state
            frames.append(AnimationFrame(
                state=state,
                time=total_time,
                label=f"Step {i}: {state.name}"
            ))
            total_time += 1.0 / self.config.fps
            
            # Add interpolated frames to next state
            if i < len(states) - 1:
                start_coords = state.get_bloch_coordinates(qubit_index)
                end_coords = states[i + 1].get_bloch_coordinates(qubit_index)
                
                for j in range(1, frames_between):
                    t = j / frames_between
                    t_eased = self._easing(t, self.config.easing)
                    
                    interp_coords = self._interpolate_bloch(
                        start_coords, end_coords, t_eased, "geodesic"
                    )
                    
                    # Create interpolated state
                    interp_state = self._create_state_from_bloch(interp_coords)
                    interp_state.name = f"transition_{i}_{j}"
                    
                    frames.append(AnimationFrame(
                        state=interp_state,
                        time=total_time,
                        label=""
                    ))
                    total_time += 1.0 / self.config.fps
        
        return frames
    
    def _create_state_from_bloch(self, coords: BlochCoordinates) -> QuantumState:
        """Create a quantum state from Bloch coordinates."""
        theta, phi = coords.theta, coords.phi
        
        # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        
        sv = np.array([alpha, beta], dtype=complex)
        return QuantumState(num_qubits=1, state_vector=sv, name="interpolated")
    
    def animate_bloch_sphere(self,
                            states: List[QuantumState],
                            qubit_index: int = 0,
                            output_path: Optional[str] = None,
                            show: bool = True) -> Any:
        """
        Create animated Bloch sphere visualization.
        
        Args:
            states: List of quantum states
            qubit_index: Which qubit to animate
            output_path: Path to save animation (GIF or MP4)
            show: Whether to display animation
            
        Returns:
            Animation object
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        
        # Setup figure
        plt.style.use('dark_background' if 'dark' in self.theme.name.value else 'default')
        fig = plt.figure(figsize=(10, 10), facecolor=self.theme.colors.background)
        ax = fig.add_subplot(111, projection='3d', facecolor=self.theme.colors.background)
        ax.set_axis_off()
        
        # Pre-compute all Bloch coordinates
        all_coords = [s.get_bloch_coordinates(qubit_index) for s in states]
        
        # Create interpolated frames
        frames = self.create_evolution_frames(states, qubit_index, frames_between=5)
        coords_list = [f.state.get_bloch_coordinates(0) for f in frames]
        
        # Draw static sphere elements
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Initialize plot elements
        sphere_surface = ax.plot_surface(
            x_sphere, y_sphere, z_sphere,
            color=self.theme.colors.bloch_sphere,
            alpha=self.theme.sphere_alpha,
            linewidth=0
        )
        
        # State vector (will be updated)
        vector_quiver = ax.quiver(0, 0, 0, 0, 0, 1,
                                  color=self.theme.colors.bloch_vector,
                                  arrow_length_ratio=0.15,
                                  linewidth=3)
        
        point_scatter = ax.scatter([0], [0], [1],
                                   color=self.theme.colors.bloch_vector,
                                   s=100)
        
        # Trail (history of positions)
        trail_x, trail_y, trail_z = [], [], []
        trail_line, = ax.plot([], [], [], color=self.theme.colors.bloch_vector,
                             alpha=0.3, linewidth=1)
        
        # Title text
        title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes,
                              color=self.theme.colors.text,
                              fontsize=14, ha='center')
        
        # Info text
        info_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                             color=self.theme.colors.text_secondary,
                             fontsize=10, va='top',
                             fontfamily=self.theme.font_family)
        
        # Set limits
        limit = 1.3
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.set_box_aspect([1, 1, 1])
        
        # Draw axes labels
        ax.text(1.2, 0, 0, '|x+⟩', color=self.theme.colors.state_0)
        ax.text(0, 1.2, 0, '|y+⟩', color=self.theme.colors.state_1)
        ax.text(0, 0, 1.2, '|0⟩', color=self.theme.colors.text)
        ax.text(0, 0, -1.2, '|1⟩', color=self.theme.colors.text)
        
        def init():
            """Initialize animation."""
            return [point_scatter, trail_line, title_text, info_text]
        
        def update(frame_idx):
            """Update animation frame."""
            nonlocal vector_quiver, trail_x, trail_y, trail_z
            
            frame = frames[frame_idx]
            coords = coords_list[frame_idx]
            
            # Update vector
            vector_quiver.remove()
            vector_quiver = ax.quiver(0, 0, 0, coords.x, coords.y, coords.z,
                                     color=self.theme.colors.bloch_vector,
                                     arrow_length_ratio=0.15,
                                     linewidth=3)
            
            # Update point
            point_scatter._offsets3d = ([coords.x], [coords.y], [coords.z])
            
            # Update trail
            trail_x.append(coords.x)
            trail_y.append(coords.y)
            trail_z.append(coords.z)
            
            # Keep trail limited
            max_trail = self.theme.trail_length
            if len(trail_x) > max_trail:
                trail_x.pop(0)
                trail_y.pop(0)
                trail_z.pop(0)
            
            trail_line.set_data_3d(trail_x, trail_y, trail_z)
            
            # Update text
            if frame.label:
                title_text.set_text(frame.label)
            
            info = f"θ = {coords.theta:.2f}\nφ = {coords.phi:.2f}"
            info_text.set_text(info)
            
            # Rotate view slightly for 3D effect
            ax.view_init(elev=self.theme.elevation, 
                        azim=self.theme.azimuth + frame_idx * 0.5)
            
            return [point_scatter, trail_line, title_text, info_text, vector_quiver]
        
        # Create animation
        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=len(frames),
            interval=1000 / self.config.fps,
            blit=False,
            repeat=self.config.loop
        )
        
        # Save if path provided
        if output_path:
            if output_path.endswith('.gif'):
                anim.save(output_path, writer='pillow', fps=self.config.fps)
            elif output_path.endswith('.mp4'):
                anim.save(output_path, writer='ffmpeg', fps=self.config.fps)
            print(f"Animation saved to: {output_path}")
        
        if show:
            plt.show()
        
        return anim
    
    def animate_probability_histogram(self,
                                     states: List[QuantumState],
                                     output_path: Optional[str] = None,
                                     show: bool = True) -> Any:
        """
        Animate probability distribution evolution.
        
        Args:
            states: List of quantum states
            output_path: Path to save animation
            show: Whether to display
            
        Returns:
            Animation object
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        plt.style.use('dark_background' if 'dark' in self.theme.name.value else 'default')
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=self.theme.colors.background)
        ax.set_facecolor(self.theme.colors.surface)
        
        n = states[0].num_qubits
        num_states = 2 ** n
        basis_labels = [format(i, f'0{n}b') for i in range(num_states)]
        x = np.arange(num_states)
        
        # Initial probabilities
        initial_probs = states[0].get_probabilities()
        bars = ax.bar(x, initial_probs, color=self.theme.colors.primary, alpha=0.8)
        
        ax.set_xlim(-0.5, num_states - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(basis_labels, fontsize=10, rotation=45 if n > 3 else 0)
        ax.set_ylabel('Probability', fontsize=12, color=self.theme.colors.text)
        ax.set_xlabel('Basis State', fontsize=12, color=self.theme.colors.text)
        
        title = ax.set_title('', fontsize=14, color=self.theme.colors.text)
        
        # Color by probability
        cmap = self.theme.get_cmap('quantum')
        
        def update(frame_idx):
            state = states[frame_idx]
            probs = state.get_probabilities()
            
            for bar, prob in zip(bars, probs):
                bar.set_height(prob)
                bar.set_color(cmap(prob))
            
            title.set_text(f'Step {frame_idx + 1}/{len(states)}: {state.name}')
            
            return list(bars) + [title]
        
        anim = FuncAnimation(
            fig, update,
            frames=len(states),
            interval=500,
            blit=False,
            repeat=self.config.loop
        )
        
        if output_path:
            anim.save(output_path, writer='pillow' if output_path.endswith('.gif') else 'ffmpeg',
                     fps=2)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return anim
    
    def terminal_animation(self,
                          states: List[QuantumState],
                          qubit_index: int = 0,
                          delay: float = 0.1):
        """
        Display animation in terminal using ASCII art.
        
        Args:
            states: List of quantum states
            qubit_index: Which qubit to animate
            delay: Delay between frames in seconds
        """
        import os
        
        for i, state in enumerate(states):
            # Clear terminal
            os.system('cls' if os.name == 'nt' else 'clear')
            
            coords = state.get_bloch_coordinates(qubit_index)
            
            # Create ASCII visualization
            print(self._ascii_bloch(coords, state.name, i + 1, len(states)))
            
            time_module.sleep(delay)
    
    def _ascii_bloch(self, coords: BlochCoordinates, name: str, 
                     step: int, total: int) -> str:
        """Generate ASCII Bloch sphere representation."""
        # Determine approximate position on a 2D projection
        # Map (x, y, z) to a character grid
        
        width, height = 40, 20
        center_x, center_y = width // 2, height // 2
        
        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Draw sphere outline (ellipse)
        for angle in np.linspace(0, 2 * np.pi, 60):
            px = int(center_x + 15 * np.cos(angle))
            py = int(center_y + 8 * np.sin(angle))
            if 0 <= px < width and 0 <= py < height:
                grid[py][px] = '·'
        
        # Draw axes
        for i in range(-15, 16):
            if 0 <= center_x + i < width:
                grid[center_y][center_x + i] = '─'
            if 0 <= center_y + i // 2 < height:
                grid[center_y + i // 2][center_x] = '│'
        
        grid[center_y][center_x] = '┼'
        
        # Draw state vector position (projected)
        # Simple orthographic projection
        px = int(center_x + 15 * coords.x)
        py = int(center_y - 8 * coords.z)  # Flip z for display
        
        if 0 <= px < width and 0 <= py < height:
            grid[py][px] = '●'
        
        # Draw projection line to center
        steps = max(abs(px - center_x), abs(py - center_y))
        if steps > 0:
            for s in range(1, steps):
                lx = int(center_x + (px - center_x) * s / steps)
                ly = int(center_y + (py - center_y) * s / steps)
                if 0 <= lx < width and 0 <= ly < height:
                    if grid[ly][lx] == ' ':
                        grid[ly][lx] = '·'
        
        # Convert grid to string
        result = []
        result.append("╔" + "═" * (width + 2) + "╗")
        result.append(f"║ 🔮 QuantumViz - Bloch Sphere Animation {' ' * (width - 37)}║")
        result.append("╠" + "═" * (width + 2) + "╣")
        
        # Add |0⟩ label
        result.append(f"║ {'|0⟩'.center(width)} ║")
        
        for row in grid:
            result.append("║ " + ''.join(row) + " ║")
        
        # Add |1⟩ label
        result.append(f"║ {'|1⟩'.center(width)} ║")
        
        result.append("╠" + "═" * (width + 2) + "╣")
        result.append(f"║ State: {name[:30].ljust(30)} Step: {step}/{total} {' ' * (width - 50)}║")
        result.append(f"║ θ = {coords.theta:6.3f} rad  φ = {coords.phi:6.3f} rad {' ' * (width - 38)}║")
        result.append(f"║ (x,y,z) = ({coords.x:5.2f}, {coords.y:5.2f}, {coords.z:5.2f}) {' ' * (width - 37)}║")
        result.append("╚" + "═" * (width + 2) + "╝")
        
        return '\n'.join(result)
    
    def progress_bar(self, current: int, total: int, width: int = 40, 
                    prefix: str = "Progress") -> str:
        """Generate a progress bar string."""
        percent = current / total
        filled = int(width * percent)
        bar = '█' * filled + '░' * (width - filled)
        return f"{prefix}: [{bar}] {current}/{total} ({percent*100:.1f}%)"
