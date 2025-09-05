"""Production environment configuration."""

from .base import *

# Production settings
DEBUG = False
LOG_LEVEL = "INFO"

# Production model settings
MEMORY_DIM = DEFAULT_MEMORY_DIM
TIME_DIM = DEFAULT_TIME_DIM  
EDGE_DIM = DEFAULT_EDGE_DIM

# Production service settings
MAX_NODES = DEFAULT_MAX_NODES
EMBED_BUDGET_MS = DEFAULT_EMBED_BUDGET_MS