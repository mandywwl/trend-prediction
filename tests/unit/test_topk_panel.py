"""Tests for TopK dashboard panel functionality."""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

from utils.path_utils import find_repo_root

# Add paths for imports
repo_root = find_repo_root()
dashboard_path = repo_root / "dashboard"
sys.path.insert(0, str(dashboard_path))

from dashboard.components.topk import _load_predictions_cache, _format_countdown, _get_latest_predictions


def test_load_predictions_cache():
    """Test loading predictions cache from JSON file."""
    # Create a temporary cache file
    cache_data = {
        "last_updated": "2024-01-15T10:30:00Z",
        "items": [
            {
                "t_iso": "2024-01-15T10:00:00Z",
                "topk": [
                    {"topic_id": 42, "score": 0.95},
                    {"topic_id": 17, "score": 0.87}
                ]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cache_data, f)
        temp_file = f.name
    
    try:
        # Load the cache
        loaded_cache = _load_predictions_cache(temp_file)
        
        # Verify structure
        assert loaded_cache is not None
        assert loaded_cache["last_updated"] == "2024-01-15T10:30:00Z"
        assert len(loaded_cache["items"]) == 1
        assert len(loaded_cache["items"][0]["topk"]) == 2
        assert loaded_cache["items"][0]["topk"][0]["topic_id"] == 42
        assert loaded_cache["items"][0]["topk"][0]["score"] == 0.95
        
    finally:
        Path(temp_file).unlink()


def test_load_predictions_cache_missing_file():
    """Test loading cache when file doesn't exist."""
    cache = _load_predictions_cache("nonexistent_file.json")
    assert cache is None


def test_format_countdown():
    """Test countdown timer formatting."""
    # Test with 1 hour 30 minutes remaining
    now = datetime.now(timezone.utc)
    pred_time = now - timedelta(minutes=30)  # 30 minutes ago
    delta_hours = 2  # 2 hour window
    
    countdown = _format_countdown(delta_hours, pred_time.isoformat())
    
    # Should show approximately 1h 30m remaining
    assert "1h" in countdown
    assert "30m" in countdown or "29m" in countdown  # Allow for timing differences
    assert "🕐" in countdown


def test_format_countdown_frozen():
    """Test countdown when prediction is past freeze time."""
    # Test prediction that should be frozen
    now = datetime.now(timezone.utc)
    pred_time = now - timedelta(hours=3)  # 3 hours ago
    delta_hours = 2  # 2 hour window (so it's been frozen for 1 hour)
    
    countdown = _format_countdown(delta_hours, pred_time.isoformat())
    
    assert countdown == "⏰ Frozen"


def test_get_latest_predictions():
    """Test getting the most recent predictions from cache."""
    cache = {
        "last_updated": "2024-01-15T10:30:00Z",
        "items": [
            {
                "t_iso": "2024-01-15T09:00:00Z",  # Older
                "topk": [{"topic_id": 1, "score": 0.5}]
            },
            {
                "t_iso": "2024-01-15T10:00:00Z",  # Newer
                "topk": [{"topic_id": 42, "score": 0.95}]
            },
            {
                "t_iso": "2024-01-15T08:00:00Z",  # Oldest
                "topk": [{"topic_id": 2, "score": 0.3}]
            }
        ]
    }
    
    latest = _get_latest_predictions(cache)
    
    assert latest is not None
    assert latest["t_iso"] == "2024-01-15T10:00:00Z"
    assert latest["topk"][0]["topic_id"] == 42
    assert latest["topk"][0]["score"] == 0.95


def test_get_latest_predictions_empty_cache():
    """Test getting predictions from empty cache."""
    empty_cache = {"last_updated": "2024-01-15T10:30:00Z", "items": []}
    
    latest = _get_latest_predictions(empty_cache)
    assert latest is None
    
    # Test with None cache
    latest = _get_latest_predictions(None)
    assert latest is None


if __name__ == "__main__":
    test_load_predictions_cache()
    test_load_predictions_cache_missing_file()
    test_format_countdown()
    test_format_countdown_frozen()
    test_get_latest_predictions()
    test_get_latest_predictions_empty_cache()
    print("All TopK panel tests passed!")

