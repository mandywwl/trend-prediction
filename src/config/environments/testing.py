"""Testing environment configuration."""

from ..base import *

# Test settings
DEBUG = True
LOG_LEVEL = "WARNING"

# Test model settings - smaller for faster tests
MEMORY_DIM = 10
TIME_DIM = 5
EDGE_DIM = 32

# Test service settings 
MAX_NODES = 100  # Small for tests
EMBED_BUDGET_MS = 10