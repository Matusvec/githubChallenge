"""
Configuration Management
========================

Handles application configuration and settings.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import os


@dataclass
class VisualizationSettings:
    """Visualization configuration."""
    default_theme: str = "quantum_dark"
    figure_size: tuple = (10, 8)
    dpi: int = 100
    animation_fps: int = 30
    show_grid: bool = True
    use_latex: bool = False


@dataclass
class SimulationSettings:
    """Simulation configuration."""
    default_shots: int = 1000
    max_qubits: int = 20
    use_gpu: bool = False
    precision: str = "double"  # "single" or "double"
    seed: Optional[int] = None


@dataclass
class AISettings:
    """AI/Copilot configuration."""
    enable_copilot: bool = True
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.3
    cache_responses: bool = True


@dataclass 
class Config:
    """
    Application configuration.
    
    Configuration can be loaded from:
    1. Default values
    2. Config file (~/.quantumviz/config.json)
    3. Environment variables (QUANTUMVIZ_*)
    
    Example:
        >>> config = Config.load()
        >>> config.visualization.default_theme = "cyberpunk"
        >>> config.save()
    """
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    ai: AISettings = field(default_factory=AISettings)
    
    # Paths
    cache_dir: Optional[str] = None
    export_dir: Optional[str] = None
    
    @classmethod
    def get_config_path(cls) -> Path:
        """Get the configuration file path."""
        config_dir = Path.home() / ".quantumviz"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.json"
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """
        Load configuration from file.
        
        Args:
            path: Config file path (uses default if not provided)
            
        Returns:
            Config instance
        """
        path = path or cls.get_config_path()
        
        config = cls()
        
        # Load from file if exists
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                config = cls._from_dict(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config from {path}: {e}")
        
        # Override with environment variables
        config._load_env_overrides()
        
        return config
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        path = path or self.get_config_path()
        
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        
        if "visualization" in data:
            config.visualization = VisualizationSettings(**data["visualization"])
        if "simulation" in data:
            config.simulation = SimulationSettings(**data["simulation"])
        if "ai" in data:
            config.ai = AISettings(**data["ai"])
        if "cache_dir" in data:
            config.cache_dir = data["cache_dir"]
        if "export_dir" in data:
            config.export_dir = data["export_dir"]
        
        return config
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "visualization": asdict(self.visualization),
            "simulation": asdict(self.simulation),
            "ai": {k: v for k, v in asdict(self.ai).items() if k != "api_key"},  # Don't save API key
            "cache_dir": self.cache_dir,
            "export_dir": self.export_dir,
        }
    
    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        # AI settings
        if api_key := os.environ.get("QUANTUMVIZ_API_KEY") or os.environ.get("COPILOT_API_KEY"):
            self.ai.api_key = api_key
        
        if os.environ.get("QUANTUMVIZ_DISABLE_COPILOT"):
            self.ai.enable_copilot = False
        
        # Visualization
        if theme := os.environ.get("QUANTUMVIZ_THEME"):
            self.visualization.default_theme = theme
        
        # Simulation
        if shots := os.environ.get("QUANTUMVIZ_DEFAULT_SHOTS"):
            self.simulation.default_shots = int(shots)
        
        if seed := os.environ.get("QUANTUMVIZ_SEED"):
            self.simulation.seed = int(seed)
        
        if os.environ.get("QUANTUMVIZ_USE_GPU"):
            self.simulation.use_gpu = True
    
    def get_cache_path(self, filename: str) -> Path:
        """Get path for a cache file."""
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
        else:
            cache_path = Path.home() / ".quantumviz" / "cache"
        
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path / filename
    
    def get_export_path(self, filename: str) -> Path:
        """Get path for an export file."""
        if self.export_dir:
            export_path = Path(self.export_dir)
        else:
            export_path = Path.cwd()
        
        export_path.mkdir(parents=True, exist_ok=True)
        return export_path / filename
