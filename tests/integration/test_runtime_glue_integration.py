"""Integration test for runtime glue with mock streams."""

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from service.runtime_glue import RuntimeGlue, RuntimeConfig, mock_event_stream


class MockEventHandler:
    """Mock event handler that simulates real prediction scoring."""
    
    def __init__(self):
        self.events_processed = []
        self.prediction_counter = 0
    
    def on_event(self, event):
        self.events_processed.append(event)
        self.prediction_counter += 1
        
        # Simulate varying prediction scores
        base_score = 0.5 + (self.prediction_counter % 10) * 0.05
        return {
            f'trend_topic_{self.prediction_counter % 5}': base_score,
            f'emerging_topic_{self.prediction_counter % 3}': base_score + 0.2,
            f'declining_topic_{self.prediction_counter % 7}': max(0.1, base_score - 0.3),
        }


def test_runtime_glue_integration():
    """Test full runtime glue integration with mock streams."""
    print("Starting runtime glue integration test...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metrics_dir = temp_path / "metrics"
        cache_path = temp_path / "predictions_cache.json"
        
        # Create configuration
        config = RuntimeConfig(
            delta_hours=1,  # Shorter for testing
            window_min=30,
            k_default=5,
            update_interval_sec=2,  # Update every 2 seconds for testing
            metrics_snapshot_dir=str(metrics_dir),
            predictions_cache_path=str(cache_path)
        )
        
        # Create mock handler
        handler = MockEventHandler()
        
        # Create runtime glue
        glue = RuntimeGlue(handler, config)
        
        print(f"Configuration: {config.__dict__}")
        print("Processing mock event stream...")
        
        # Process a small number of events with forced metrics updates
        events_processed = 0
        start_time = time.time()
        
        for event in mock_event_stream(n_events=10, delay=0.1):
            # Process event
            scores = glue.event_handler.on_event(event)
            glue._record_event_for_metrics(event, scores)
            events_processed += 1
            
            # Force metrics update every few events for testing
            if events_processed % 3 == 0:
                glue._update_metrics_and_cache()
                print(f"Processed {events_processed} events, updated metrics")
            
            # Break after reasonable time to avoid long test
            if time.time() - start_time > 5:
                break
        
        # Final metrics update
        glue._update_metrics_and_cache()
        
        print(f"\nTest completed. Processed {events_processed} events.")
        print(f"Total events handled: {len(handler.events_processed)}")
        
        # Verify outputs
        print("\nVerifying outputs...")
        
        # Check metrics directory
        assert metrics_dir.exists(), "Metrics directory should exist"
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        print(f"Created {len(metrics_files)} metrics files")
        assert len(metrics_files) > 0, "Should have created at least one metrics file"
        
        # Check latest metrics file
        latest_metrics_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        with open(latest_metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        print(f"Latest metrics file: {latest_metrics_file.name}")
        print(f"Metrics content keys: {list(metrics_data.keys())}")
        
        assert 'precision_at_k' in metrics_data
        assert 'latency' in metrics_data
        assert 'meta' in metrics_data
        assert metrics_data['meta']['service'] == 'runtime_glue'
        
        # Check cache file
        assert cache_path.exists(), "Predictions cache should exist"
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        print(f"Cache items: {len(cache_data.get('items', []))}")
        assert 'last_updated' in cache_data
        assert 'items' in cache_data
        assert len(cache_data['items']) > 0, "Should have cached some predictions"
        
        # Verify cache item structure
        cache_item = cache_data['items'][0]
        assert 't_iso' in cache_item
        assert 'topk' in cache_item
        print(f"Sample cache item: {cache_item}")
        
        # Verify predictions format
        for prediction in cache_item['topk']:
            assert 'topic_id' in prediction
            assert 'score' in prediction
            assert isinstance(prediction['topic_id'], int)
            assert isinstance(prediction['score'], (int, float))
        
        print("✅ All verifications passed!")
        
        return {
            'events_processed': events_processed,
            'metrics_files': len(metrics_files),
            'cache_items': len(cache_data['items']),
            'config_used': config.__dict__
        }


if __name__ == "__main__":
    result = test_runtime_glue_integration()
    print(f"\nIntegration test results: {result}")