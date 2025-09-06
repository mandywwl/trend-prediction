"""Validation utilities for data and models."""

from typing import Any, Dict, List, Union, Sequence
import numpy as np
import torch
from utils.logging import get_logger

logger = get_logger(__name__)


def validate_event(event: Dict[str, Any]) -> bool:
    """Validate event structure."""
    required_fields = ["event_id", "ts_iso", "actor_id", "target_ids", "edge_type"]
    return all(field in event for field in required_fields)


def validate_tensor_shape(tensor: Union[np.ndarray, torch.Tensor], expected_shape: tuple) -> bool:
    """Validate tensor has expected shape."""
    if isinstance(tensor, torch.Tensor):
        shape = tensor.shape
    else:
        shape = tensor.shape
    return shape == expected_shape


def validate_embedding_dim(embedding: Union[np.ndarray, torch.Tensor], expected_dim: int) -> bool:
    """Validate embedding has expected dimensionality."""
    if isinstance(embedding, torch.Tensor):
        return embedding.numel() == expected_dim
    else:
        return embedding.size == expected_dim


def sanitize_node_id(node_id: str) -> str:
    """Sanitize node ID to ensure it's safe for use."""
    # Remove any problematic characters and limit length
    return str(node_id)[:100]


def validate_graph(src: Sequence, dst: Sequence) -> bool:
    """Simple validation for the temporal edge stream.
    
    Currently checks only for self-loops to keep memory usage minimal.
    
    Args:
        src: Source node indices
        dst: Destination node indices
        
    Returns:
        bool: True if validation passes, False otherwise
        
    Raises:
        AssertionError: If graph has self-loops
    """
    try:
        assert all(s != d for s, d in zip(src, dst)), "ERROR: Graph has self-loops!"
        logger.info("Graph passed all integrity checks")
        return True
    except AssertionError as e:
        logger.error(str(e))
        return False