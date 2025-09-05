"""Configuration models and data structures."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for TGN model parameters."""
    memory_dim: int = 100
    time_dim: int = 10
    edge_dim: int = 768
    max_nodes: int = 10000
    device: str = "cpu"
    checkpoint_path: Optional[str] = None

@dataclass 
class ServiceConfig:
    """Configuration for service layer."""
    embed_budget_ms: int = 50
    log_latencies: bool = True
    log_dir: str = "datasets"

@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline."""
    batch_size: int = 32
    max_workers: int = 4
    buffer_size: int = 1000