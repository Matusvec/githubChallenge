"""
Visualization Themes
====================

Beautiful color schemes and styling for quantum visualizations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ThemeName(Enum):
    """Available visualization themes."""
    QUANTUM_DARK = "quantum_dark"
    QUANTUM_LIGHT = "quantum_light"
    CYBERPUNK = "cyberpunk"
    SCIENTIFIC = "scientific"
    MATRIX = "matrix"
    AURORA = "aurora"
    SUNSET = "sunset"


@dataclass
class ColorPalette:
    """Color palette for visualizations."""
    primary: str = "#00D4FF"           # Main accent color
    secondary: str = "#FF00FF"          # Secondary accent
    tertiary: str = "#FFD700"           # Tertiary accent
    background: str = "#0a0a0f"         # Background color
    surface: str = "#1a1a2e"            # Surface/card color
    text: str = "#FFFFFF"               # Text color
    text_secondary: str = "#888888"     # Secondary text
    grid: str = "#333366"               # Grid lines
    positive: str = "#00FF88"           # Positive values
    negative: str = "#FF4444"           # Negative values
    neutral: str = "#888888"            # Neutral elements
    
    # Quantum state colors
    state_0: str = "#00D4FF"            # |0⟩ state
    state_1: str = "#FF00FF"            # |1⟩ state
    superposition: str = "#FFD700"      # Superposition
    entangled: str = "#FF6B35"          # Entangled states
    
    # Bloch sphere colors
    bloch_sphere: str = "#1a1a2e"       # Sphere surface
    bloch_wireframe: str = "#333366"    # Wireframe
    bloch_vector: str = "#00FF88"       # State vector
    bloch_axes: str = "#666666"         # Axis lines
    
    # Gradient colors (for heatmaps, animations)
    gradient_start: str = "#0066FF"
    gradient_mid: str = "#FF00FF"
    gradient_end: str = "#FF6600"


@dataclass
class VisualizationTheme:
    """Complete visualization theme configuration."""
    
    name: ThemeName
    colors: ColorPalette
    
    # Typography
    font_family: str = "monospace"
    title_size: int = 16
    label_size: int = 12
    tick_size: int = 10
    
    # Bloch sphere settings
    sphere_resolution: int = 50
    sphere_alpha: float = 0.1
    wireframe_alpha: float = 0.3
    vector_width: float = 3.0
    point_size: float = 100
    
    # Animation settings
    fps: int = 30
    trail_length: int = 20
    trail_alpha: float = 0.5
    glow_effect: bool = True
    particle_effects: bool = True
    
    # Circuit drawing
    gate_width: float = 0.8
    gate_height: float = 0.6
    wire_spacing: float = 1.0
    
    # 3D settings
    elevation: float = 20.0
    azimuth: float = 45.0
    zoom: float = 1.0
    
    def get_cmap(self, name: str = "quantum"):
        """Get a matplotlib colormap for the theme."""
        import matplotlib.colors as mcolors
        
        if name == "quantum":
            colors = [self.colors.gradient_start, self.colors.gradient_mid, self.colors.gradient_end]
        elif name == "probability":
            colors = [self.colors.background, self.colors.primary, self.colors.secondary]
        elif name == "phase":
            colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#FF00FF', '#0000FF']
        else:
            colors = [self.colors.primary, self.colors.secondary]
        
        return mcolors.LinearSegmentedColormap.from_list(name, colors)
    
    @classmethod
    def quantum_dark(cls) -> "VisualizationTheme":
        """Dark theme with quantum-inspired colors."""
        return cls(
            name=ThemeName.QUANTUM_DARK,
            colors=ColorPalette(
                primary="#00D4FF",
                secondary="#FF00FF",
                tertiary="#FFD700",
                background="#0a0a0f",
                surface="#1a1a2e",
                text="#FFFFFF",
                grid="#333366",
                bloch_vector="#00FF88",
            )
        )
    
    @classmethod
    def quantum_light(cls) -> "VisualizationTheme":
        """Light theme for presentations."""
        return cls(
            name=ThemeName.QUANTUM_LIGHT,
            colors=ColorPalette(
                primary="#0066CC",
                secondary="#CC00CC",
                tertiary="#CC9900",
                background="#FFFFFF",
                surface="#F5F5F5",
                text="#333333",
                text_secondary="#666666",
                grid="#CCCCCC",
                bloch_sphere="#F0F0F0",
                bloch_wireframe="#CCCCCC",
                bloch_vector="#00AA66",
                bloch_axes="#999999",
            )
        )
    
    @classmethod
    def cyberpunk(cls) -> "VisualizationTheme":
        """Neon cyberpunk aesthetic."""
        return cls(
            name=ThemeName.CYBERPUNK,
            colors=ColorPalette(
                primary="#FF0080",
                secondary="#00FFFF",
                tertiary="#FFE600",
                background="#0D0221",
                surface="#1A0533",
                text="#FFFFFF",
                grid="#4A0080",
                state_0="#00FFFF",
                state_1="#FF0080",
                superposition="#FFE600",
                bloch_sphere="#1A0533",
                bloch_wireframe="#4A0080",
                bloch_vector="#00FF00",
                gradient_start="#FF0080",
                gradient_mid="#8000FF",
                gradient_end="#00FFFF",
            ),
            glow_effect=True,
            particle_effects=True,
        )
    
    @classmethod
    def scientific(cls) -> "VisualizationTheme":
        """Clean scientific publication style."""
        return cls(
            name=ThemeName.SCIENTIFIC,
            colors=ColorPalette(
                primary="#2E86AB",
                secondary="#A23B72",
                tertiary="#F18F01",
                background="#FFFFFF",
                surface="#FAFAFA",
                text="#1A1A1A",
                text_secondary="#666666",
                grid="#E0E0E0",
                bloch_sphere="#F5F5F5",
                bloch_wireframe="#CCCCCC",
                bloch_vector="#2E86AB",
                bloch_axes="#999999",
            ),
            font_family="serif",
            glow_effect=False,
            particle_effects=False,
        )
    
    @classmethod
    def matrix(cls) -> "VisualizationTheme":
        """Matrix/hacker green aesthetic."""
        return cls(
            name=ThemeName.MATRIX,
            colors=ColorPalette(
                primary="#00FF00",
                secondary="#00CC00",
                tertiary="#88FF88",
                background="#000000",
                surface="#0A0A0A",
                text="#00FF00",
                text_secondary="#008800",
                grid="#003300",
                state_0="#00FF00",
                state_1="#00CC00",
                superposition="#88FF88",
                bloch_sphere="#0A0A0A",
                bloch_wireframe="#003300",
                bloch_vector="#00FF00",
                gradient_start="#001100",
                gradient_mid="#00FF00",
                gradient_end="#AAFFAA",
            ),
            font_family="monospace",
        )
    
    @classmethod
    def aurora(cls) -> "VisualizationTheme":
        """Northern lights inspired theme."""
        return cls(
            name=ThemeName.AURORA,
            colors=ColorPalette(
                primary="#00FF88",
                secondary="#00CCFF",
                tertiary="#FF88FF",
                background="#0A1628",
                surface="#132238",
                text="#FFFFFF",
                grid="#1E3A5F",
                state_0="#00FF88",
                state_1="#00CCFF",
                superposition="#FF88FF",
                bloch_sphere="#132238",
                bloch_wireframe="#1E3A5F",
                bloch_vector="#00FF88",
                gradient_start="#00FF88",
                gradient_mid="#00CCFF",
                gradient_end="#FF88FF",
            ),
            glow_effect=True,
        )
    
    @classmethod
    def sunset(cls) -> "VisualizationTheme":
        """Warm sunset colors."""
        return cls(
            name=ThemeName.SUNSET,
            colors=ColorPalette(
                primary="#FF6B35",
                secondary="#F7C59F",
                tertiary="#2EC4B6",
                background="#1A0F0A",
                surface="#2D1810",
                text="#FFFFFF",
                grid="#4A2820",
                state_0="#FF6B35",
                state_1="#F7C59F",
                superposition="#2EC4B6",
                bloch_sphere="#2D1810",
                bloch_wireframe="#4A2820",
                bloch_vector="#FF6B35",
                gradient_start="#FF6B35",
                gradient_mid="#F7C59F",
                gradient_end="#2EC4B6",
            ),
        )
    
    @classmethod
    def get_theme(cls, name: str) -> "VisualizationTheme":
        """Get theme by name."""
        themes = {
            "quantum_dark": cls.quantum_dark,
            "quantum_light": cls.quantum_light,
            "cyberpunk": cls.cyberpunk,
            "scientific": cls.scientific,
            "matrix": cls.matrix,
            "aurora": cls.aurora,
            "sunset": cls.sunset,
        }
        
        if name.lower() not in themes:
            raise ValueError(f"Unknown theme: {name}. Available: {list(themes.keys())}")
        
        return themes[name.lower()]()
    
    @classmethod
    def list_themes(cls) -> List[str]:
        """List all available themes."""
        return [t.value for t in ThemeName]


# ASCII art elements for terminal visualization
ASCII_ELEMENTS = {
    "sphere_small": """
       ___
     /     \\
    |   ●   |
     \\_____/
    """,
    
    "sphere_medium": """
         _____
       /       \\
      /         \\
     |     ●     |
      \\         /
       \\_______/
    """,
    
    "qubit_0": "│0⟩",
    "qubit_1": "│1⟩",
    "qubit_plus": "│+⟩",
    "qubit_minus": "│-⟩",
    
    "gate_h": "┤ H ├",
    "gate_x": "┤ X ├",
    "gate_y": "┤ Y ├",
    "gate_z": "┤ Z ├",
    "gate_cnot_ctrl": "──●──",
    "gate_cnot_targ": "──⊕──",
    
    "wire": "─────",
    "barrier": "│",
    "measure": "─M─",
}
