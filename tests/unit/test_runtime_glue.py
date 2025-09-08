"""Tests for the runtime glue component."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from service.runtime_glue import RuntimeGlue, RuntimeConfig, mock_event_stream
from config.schemas import Event


class MockEventHandler:
    """Mock event handler for testing."""
    
    def __init__(self):
        self.events_processed = []
    
    def on_event(self, event: Event) -> dict:
        self.events_processed.append(event)
        # Return mock prediction scores
        return {
            'topic_1': 0.8,
            'topic_2': 0.6,
            'topic_3': 0.4
        }


def test_runtime_config_defaults():
    """Test that RuntimeConfig uses correct defaults."""
    config = RuntimeConfig()
    
    assert config.delta_hours == 2  # DELTA_HOURS
    assert config.window_min == 60  # WINDOW_MIN
    assert config.k_default == 5  # K_DEFAULT
    assert config.update_interval_sec == 60


def test_runtime_config_yaml_override():
    """Test YAML config override functionality."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_content = """
runtime:
  delta_hours: 3
  window_min: 90
  k_default: 10
  update_interval_sec: 30
"""
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        config = RuntimeConfig.from_yaml(yaml_path)
        assert config.delta_hours == 3
        assert config.window_min == 90
        assert config.k_default == 10
        assert config.update_interval_sec == 30
    finally:
        Path(yaml_path).unlink()


def test_runtime_config_invalid_yaml():
    """Test that invalid YAML falls back to defaults."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        yaml_path = f.name
    
    try:
        config = RuntimeConfig.from_yaml(yaml_path)
        # Should fall back to defaults
        assert config.delta_hours == 2
        assert config.window_min == 60
    finally:
        Path(yaml_path).unlink()


def test_mock_event_stream():
    """Test mock event stream generator."""
    events = list(mock_event_stream(n_events=5, delay=0))
    
    assert len(events) == 5
    for i, event in enumerate(events):
        assert event['event_id'] == f'mock_event_{i}'
        assert 'ts_iso' in event
        assert 'actor_id' in event
        assert 'target_ids' in event
        assert 'edge_type' in event
        assert 'features' in event


def test_runtime_glue_initialization():
    """Test RuntimeGlue initialization."""
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RuntimeConfig(
            metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
            predictions_cache_path=str(Path(temp_dir) / "cache.json")
        )
        
        glue = RuntimeGlue(handler, config)
        
        assert glue.event_handler == handler
        assert glue.config == config
        assert glue.precision_tracker is not None
        assert glue.metrics_writer is not None


def test_runtime_glue_processes_events():
    """Test that RuntimeGlue processes events through the handler."""
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RuntimeConfig(
            metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
            predictions_cache_path=str(Path(temp_dir) / "cache.json"),
            update_interval_sec=1  # Update every second for testing
        )
        
        glue = RuntimeGlue(handler, config)
        
        # Run with a small number of events
        events = list(mock_event_stream(n_events=3, delay=0))
        
        # Mock the stream processing to avoid infinite loop
        for event in events:
            scores = glue.event_handler.on_event(event)
            glue._record_event_for_metrics(event, scores)
        
        # Verify events were processed
        assert len(handler.events_processed) == 3
        assert handler.events_processed[0]['event_id'] == 'mock_event_0'


def test_runtime_glue_updates_metrics():
    """Test that RuntimeGlue updates metrics and cache."""
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_dir = Path(temp_dir) / "metrics"
        cache_path = Path(temp_dir) / "cache.json"
        
        config = RuntimeConfig(
            metrics_snapshot_dir=str(metrics_dir),
            predictions_cache_path=str(cache_path),
            update_interval_sec=0  # Always update for testing
        )
        
        glue = RuntimeGlue(handler, config)
        
        # Trigger metrics update
        glue._update_metrics_and_cache()
        
        # Check that metrics directory was created
        assert metrics_dir.exists()
        
        # Check that metrics files are written
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        assert len(metrics_files) > 0
        
        # Verify metrics file content
        with open(metrics_files[0], 'r') as f:
            metrics_data = json.load(f)

        assert 'precision_at_k' in metrics_data
        assert 'adaptivity' in metrics_data
        assert 'latency' in metrics_data
        assert 'meta' in metrics_data
        assert metrics_data['meta']['service'] == 'runtime_glue'


def test_runtime_glue_updates_predictions_cache():
    """Test that RuntimeGlue updates the predictions cache."""
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "cache.json"
        
        config = RuntimeConfig(
            metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
            predictions_cache_path=str(cache_path)
        )
        
        glue = RuntimeGlue(handler, config)
        
        # Add some predictions to buffer
        glue._predictions_buffer = [
            [
                {"topic_id": 1, "score": 0.8},
                {"topic_id": 2, "score": 0.6},
                {"topic_id": 3, "score": 0.4}
            ]
        ]
        
        # Trigger cache update
        glue._update_predictions_cache(datetime.now(timezone.utc))
        
        # Check that cache file was created
        assert cache_path.exists()
        
        # Verify cache content
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        assert 'last_updated' in cache_data
        assert 'items' in cache_data
        assert len(cache_data['items']) == 1
        
        cache_item = cache_data['items'][0]
        assert 't_iso' in cache_item
        assert 'topk' in cache_item
        assert len(cache_item['topk']) <= config.k_default


def test_runtime_glue_graceful_shutdown():
    """Test graceful shutdown functionality."""
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RuntimeConfig(
            metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
            predictions_cache_path=str(Path(temp_dir) / "cache.json")
        )
        
        glue = RuntimeGlue(handler, config)
        
        # Test shutdown without errors
        glue._shutdown()
        
        # Verify final metrics were written
        metrics_dir = Path(temp_dir) / "metrics"
        assert metrics_dir.exists()


def test_runtime_glue_no_magic_numbers():
    """Test that no magic numbers are used in the implementation."""
    handler = MockEventHandler()
    
    # Test with custom configuration values
    config = RuntimeConfig(
        delta_hours=4,
        window_min=120,
        k_default=15,
        update_interval_sec=90
    )
    
    glue = RuntimeGlue(handler, config)
    
    # Verify that the custom config values are used
    assert glue.precision_tracker.delta_hours == 4
    assert glue.precision_tracker.window_min == 120
    assert glue.precision_tracker.k_default == 15
    assert glue.config.update_interval_sec == 90


def test_background_timer_starts_and_stops():
    """Test that background timer can be started and stopped properly."""
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RuntimeConfig(
            metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
            predictions_cache_path=str(Path(temp_dir) / "cache.json"),
            update_interval_sec=1  # Short interval for testing
        )
        
        glue = RuntimeGlue(handler, config)
        
        # Initially no timer should be running
        assert not glue._timer_running
        assert glue._timer_thread is None
        
        # Start timer
        glue._start_background_timer()
        assert glue._timer_running
        assert glue._timer_thread is not None
        assert glue._timer_thread.is_alive()
        
        # Stop timer
        glue._stop_background_timer()
        assert not glue._timer_running
        
        # Allow thread to finish
        import time
        time.sleep(0.1)


def test_background_timer_updates_cache_without_events():
    """Test that background timer updates cache even without processing events."""
    import time
    
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "cache.json"
        
        config = RuntimeConfig(
            metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
            predictions_cache_path=str(cache_path),
            update_interval_sec=1  # Update every second for testing
        )
        
        glue = RuntimeGlue(handler, config)
        
        # Add mock predictions to buffer
        glue._predictions_buffer = [
            [
                {"topic_id": 1, "score": 0.9},
                {"topic_id": 2, "score": 0.7}
            ]
        ]
        
        try:
            # Start background timer
            glue._start_background_timer()
            
            # Wait for at least one update cycle
            time.sleep(1.5)
            
            # Cache should have been updated by background timer
            assert cache_path.exists()
            
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            assert 'last_updated' in cache_data
            assert 'items' in cache_data
            assert len(cache_data['items']) > 0
            
        finally:
            # Clean up
            glue.set_shutdown()


def test_background_timer_thread_safety():
    """Test that concurrent calls to _update_metrics_and_cache are handled safely."""
    import threading
    import time
    
    handler = MockEventHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RuntimeConfig(
            metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
            predictions_cache_path=str(Path(temp_dir) / "cache.json"),
            update_interval_sec=0  # Always allow updates for testing
        )
        
        glue = RuntimeGlue(handler, config)
        
        # Add mock predictions to buffer
        glue._predictions_buffer = [
            [
                {"topic_id": 1, "score": 0.5}
            ]
        ]
        
        # Track if any errors occur during concurrent access
        errors = []
        
        def update_metrics():
            try:
                glue._update_metrics_and_cache()
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads trying to update metrics simultaneously
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_metrics)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors due to thread safety
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])