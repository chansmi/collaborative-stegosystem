#!/usr/bin/env python3
"""
Path management for the collaborative stegosystem project.
Centralizes all path configurations and provides consistent path handling.
"""

from pathlib import Path
import os
from typing import Optional

class ProjectPaths:
    """Manages all project-related paths and ensures consistent structure."""
    
    def __init__(self, project_root: Optional[Path] = None):
        # Project root is the directory containing this file's parent
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root).resolve()
        
        # Shared resources (outside project)
        self.workspace_root = self.project_root.parent.parent
        self.shared_models = self.workspace_root / "shared_models"
        self.shared_data = self.workspace_root / "shared_data"
        
        # Project-specific directories
        self.outputs = self.project_root / "outputs"
        self.checkpoints = self.outputs / "checkpoints"
        self.loras = self.outputs / "loras"
        self.results = self.outputs / "results"
        self.logs = self.outputs / "logs"
        self.models = self.outputs / "models"
        self.experiments = self.project_root / "experiments"
        self.data = self.project_root / "data"
        
        # Ensure all directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.outputs,
            self.checkpoints,
            self.loras,
            self.results,
            self.logs,
            self.models,
            self.experiments,
            self.data
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str, model_type: str = "trained") -> Path:
        """Get path for a specific model output."""
        return self.models / f"{model_type}_{model_name}"
    
    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get path for a specific checkpoint."""
        return self.checkpoints / checkpoint_name
    
    def get_lora_path(self, lora_name: str) -> Path:
        """Get path for a specific LoRA adapter."""
        return self.loras / lora_name
    
    def get_experiment_path(self, experiment_name: str) -> Path:
        """Get path for experiment outputs."""
        return self.experiments / experiment_name
    
    def get_log_path(self, log_name: str) -> Path:
        """Get path for log files."""
        return self.logs / log_name
    
    def get_result_path(self, result_name: str) -> Path:
        """Get path for result files."""
        return self.results / result_name
    
    def get_shared_model_path(self, model_name: str) -> Path:
        """Get path for shared models."""
        return self.shared_models / model_name
    
    def get_shared_data_path(self, data_name: str) -> Path:
        """Get path for shared data."""
        return self.shared_data / data_name
    
    def get_cache_paths(self) -> dict:
        """Get all cache-related paths."""
        return {
            "hf_cache": self.workspace_root / "hf_cache",
            "transformers_cache": self.workspace_root / "transformers_cache",
            "shared_models": self.shared_models
        }
    
    def get_output_paths(self) -> dict:
        """Get all output-related paths."""
        return {
            "checkpoints": self.checkpoints,
            "loras": self.loras,
            "results": self.results,
            "logs": self.logs,
            "models": self.models,
            "experiments": self.experiments
        }
    
    def __str__(self) -> str:
        """String representation of all paths."""
        paths_info = {
            "Project Root": self.project_root,
            "Workspace Root": self.workspace_root,
            "Shared Models": self.shared_models,
            "Shared Data": self.shared_data,
            "Outputs": self.outputs,
            "Checkpoints": self.checkpoints,
            "LoRAs": self.loras,
            "Results": self.results,
            "Logs": self.logs,
            "Models": self.models,
            "Experiments": self.experiments,
            "Data": self.data
        }
        
        return "\n".join([f"{k}: {v}" for k, v in paths_info.items()])

# Global instance
project_paths = ProjectPaths()

def get_project_paths() -> ProjectPaths:
    """Get the global project paths instance."""
    return project_paths
