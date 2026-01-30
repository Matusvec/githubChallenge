"""
Export Manager
==============

Handles exporting quantum states, circuits, and visualizations
to various formats.
"""

from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for exports."""
    format: str = "json"
    include_metadata: bool = True
    precision: int = 6
    pretty_print: bool = True


class ExportManager:
    """
    Manages export of quantum data to various formats.
    
    Supported formats:
    - JSON: Full quantum state with metadata
    - CSV: Probability distributions
    - LaTeX: Circuit diagrams
    - QASM: OpenQASM circuit format
    - PNG/SVG: Visualizations
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize export manager."""
        self.config = config or ExportConfig()
    
    def export_state(
        self,
        state: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
    ) -> Path:
        """
        Export quantum state to file.
        
        Args:
            state: QuantumState object
            path: Output file path
            format: Output format (auto-detected from extension if not provided)
            
        Returns:
            Path to exported file
        """
        path = Path(path)
        format = format or path.suffix.lstrip('.').lower()
        
        if format == "json":
            return self._export_state_json(state, path)
        elif format == "csv":
            return self._export_state_csv(state, path)
        elif format == "npz":
            return self._export_state_npz(state, path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_circuit(
        self,
        circuit: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
    ) -> Path:
        """
        Export quantum circuit to file.
        
        Args:
            circuit: QuantumCircuit object
            path: Output file path
            format: Output format
            
        Returns:
            Path to exported file
        """
        path = Path(path)
        format = format or path.suffix.lstrip('.').lower()
        
        if format == "json":
            return self._export_circuit_json(circuit, path)
        elif format == "qasm":
            return self._export_circuit_qasm(circuit, path)
        elif format == "tex" or format == "latex":
            return self._export_circuit_latex(circuit, path)
        elif format == "txt":
            return self._export_circuit_ascii(circuit, path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_results(
        self,
        results: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
    ) -> Path:
        """Export simulation results."""
        path = Path(path)
        format = format or path.suffix.lstrip('.').lower()
        
        data = {
            "counts": results.counts,
            "probabilities": results.get_probabilities() if hasattr(results, 'get_probabilities') else {},
            "shots": results.shots,
        }
        
        if format == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2 if self.config.pretty_print else None)
        elif format == "csv":
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["state", "count", "probability"])
                total = sum(results.counts.values()) if results.counts else 1
                for state, count in sorted(results.counts.items()):
                    writer.writerow([state, count, count / total])
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return path
    
    def _export_state_json(self, state: Any, path: Path) -> Path:
        """Export state to JSON."""
        data = state.to_dict() if hasattr(state, 'to_dict') else {
            "num_qubits": state.num_qubits,
            "state_vector": [
                [complex(c).real, complex(c).imag] 
                for c in state.state_vector.flatten()
            ] if state.state_vector is not None else None,
            "density_matrix": [
                [[complex(c).real, complex(c).imag] for c in row]
                for row in state.density_matrix
            ] if state.density_matrix is not None else None,
            "name": state.name,
        }
        
        if self.config.include_metadata:
            data["metadata"] = {
                "exported_by": "QuantumViz",
                "version": "0.1.0",
            }
        
        with open(path, 'w') as f:
            json.dump(
                data, f, 
                indent=2 if self.config.pretty_print else None,
            )
        
        return path
    
    def _export_state_csv(self, state: Any, path: Path) -> Path:
        """Export state probabilities to CSV."""
        probs = state.get_probabilities()
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["state", "probability", "amplitude_real", "amplitude_imag"])
            
            for i, prob in enumerate(probs):
                state_str = format(i, f'0{state.num_qubits}b')
                
                if state.state_vector is not None:
                    amp = state.state_vector.flatten()[i]
                    writer.writerow([
                        state_str, 
                        round(prob, self.config.precision),
                        round(amp.real, self.config.precision),
                        round(amp.imag, self.config.precision),
                    ])
                else:
                    writer.writerow([state_str, round(prob, self.config.precision), "", ""])
        
        return path
    
    def _export_state_npz(self, state: Any, path: Path) -> Path:
        """Export state to NumPy format."""
        import numpy as np
        
        data = {
            "num_qubits": state.num_qubits,
        }
        
        if state.state_vector is not None:
            data["state_vector"] = state.state_vector
        if state.density_matrix is not None:
            data["density_matrix"] = state.density_matrix
        
        np.savez(path, **data)
        return path
    
    def _export_circuit_json(self, circuit: Any, path: Path) -> Path:
        """Export circuit to JSON."""
        data = {
            "name": circuit.name,
            "num_qubits": circuit.num_qubits,
            "operations": [
                {"gate": op[0], "qubits": list(op[1:]) if len(op) > 1 else []}
                for op in circuit.operations
            ],
            "depth": circuit.depth,
            "gate_count": circuit.gate_count,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2 if self.config.pretty_print else None)
        
        return path
    
    def _export_circuit_qasm(self, circuit: Any, path: Path) -> Path:
        """Export circuit to OpenQASM format."""
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "",
            f"qreg q[{circuit.num_qubits}];",
            f"creg c[{circuit.num_qubits}];",
            "",
        ]
        
        gate_map = {
            "h": "h",
            "x": "x",
            "y": "y",
            "z": "z",
            "s": "s",
            "t": "t",
            "sdg": "sdg",
            "tdg": "tdg",
            "rx": "rx",
            "ry": "ry",
            "rz": "rz",
            "cnot": "cx",
            "cx": "cx",
            "cz": "cz",
            "swap": "swap",
            "ccx": "ccx",
        }
        
        for op in circuit.operations:
            gate = op[0]
            
            if gate == "measure":
                qubit = op[1]
                lines.append(f"measure q[{qubit}] -> c[{qubit}];")
            elif gate in gate_map:
                qasm_gate = gate_map[gate]
                
                if gate in ["rx", "ry", "rz"] and len(op) >= 3:
                    qubit, angle = op[1], op[2]
                    lines.append(f"{qasm_gate}({angle}) q[{qubit}];")
                elif gate in ["cnot", "cx", "cz", "swap"] and len(op) >= 3:
                    q1, q2 = op[1], op[2]
                    lines.append(f"{qasm_gate} q[{q1}],q[{q2}];")
                elif gate == "ccx" and len(op) >= 4:
                    q1, q2, q3 = op[1], op[2], op[3]
                    lines.append(f"{qasm_gate} q[{q1}],q[{q2}],q[{q3}];")
                elif len(op) >= 2:
                    qubit = op[1]
                    lines.append(f"{qasm_gate} q[{qubit}];")
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        
        return path
    
    def _export_circuit_latex(self, circuit: Any, path: Path) -> Path:
        """Export circuit to LaTeX Qcircuit format."""
        from quantumviz.visualization.circuit_drawer import CircuitDrawer
        from quantumviz.visualization.themes import VisualizationTheme
        
        drawer = CircuitDrawer(VisualizationTheme.get_theme("scientific"))
        latex = drawer.to_latex(circuit)
        
        with open(path, 'w') as f:
            f.write(latex)
        
        return path
    
    def _export_circuit_ascii(self, circuit: Any, path: Path) -> Path:
        """Export circuit as ASCII art."""
        from quantumviz.visualization.circuit_drawer import CircuitDrawer
        from quantumviz.visualization.themes import VisualizationTheme
        
        drawer = CircuitDrawer(VisualizationTheme.get_theme("scientific"))
        ascii_art = drawer.to_ascii(circuit)
        
        with open(path, 'w') as f:
            f.write(ascii_art)
        
        return path
    
    def export_visualization(
        self,
        figure: Any,
        path: Union[str, Path],
        dpi: int = 300,
    ) -> Path:
        """Export matplotlib figure to file."""
        path = Path(path)
        figure.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=figure.get_facecolor())
        return path
