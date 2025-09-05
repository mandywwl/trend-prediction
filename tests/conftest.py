"""Test configuration and shared fixtures."""

import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture  
def sample_event():
    """Create a sample event for testing."""
    return {
        "event_id": "test_001",
        "ts_iso": "2024-01-01T00:00:00Z",
        "actor_id": "user_123",
        "target_ids": ["content_456"],
        "edge_type": "posted",
        "features": {}
    }