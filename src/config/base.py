"""Base configuration settings."""

import os
from pathlib import Path

# Base directory configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model configuration
DEFAULT_MEMORY_DIM = 100
DEFAULT_TIME_DIM = 10
DEFAULT_EDGE_DIM = 768

# Service configuration
DEFAULT_MAX_NODES = 10000
DEFAULT_EMBED_BUDGET_MS = 50

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)