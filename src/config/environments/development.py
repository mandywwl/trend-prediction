"""Development environment configuration."""

from .base import *

# Override for development
DEBUG = True
LOG_LEVEL = "DEBUG"

# Development-specific model settings
MEMORY_DIM = DEFAULT_MEMORY_DIM
TIME_DIM = DEFAULT_TIME_DIM
EDGE_DIM = DEFAULT_EDGE_DIM

# Development service settings
MAX_NODES = 1000  # Smaller for dev
EMBED_BUDGET_MS = DEFAULT_EMBED_BUDGET_MS