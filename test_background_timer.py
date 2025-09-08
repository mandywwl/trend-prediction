#!/usr/bin/env python3
"""
Simple test script to verify background timer functionality.
This script tests that the dashboard updates even when there are no events.
"""

import sys
import time
import tempfile
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from service.runtime_glue import RuntimeGlue, RuntimeConfig


class MockEventHandler:
    """Mock event handler for testing."""
    def on_event(self, event):
        return {"mock_topic_1": 0.8, "mock_topic_2": 0.6}


def test_background_timer():
    """Test that background timer continues to update cache even without events."""
    print("Testing background timer functionality...")
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metrics_dir = temp_path / "metrics"
        cache_path = temp_path / "predictions_cache.json"
        
        # Create config with short update interval for testing
        config = RuntimeConfig(
            metrics_snapshot_dir=str(metrics_dir),
            predictions_cache_path=str(cache_path),
            update_interval_sec=2  # Update every 2 seconds for testing
        )
        
        # Create RuntimeGlue instance
        handler = MockEventHandler()
        glue = RuntimeGlue(handler, config)
        
        print(f"Config: update_interval_sec={config.update_interval_sec}")
        print(f"Cache path: {cache_path}")
        
        try:
            # Start the background timer manually
            glue._start_background_timer()
            
            # Add some mock predictions to buffer to simulate real usage
            glue._predictions_buffer = [
                [
                    {"topic_id": 1, "score": 0.9},
                    {"topic_id": 2, "score": 0.7},
                    {"topic_id": 3, "score": 0.5}
                ]
            ]
            
            print("Started background timer. Waiting for updates...")
            
            # Wait and check for cache updates without any events
            initial_time = time.time()
            cache_updates = 0
            last_mtime = None
            
            for i in range(3):  # Check for 3 update cycles
                # Wait for update interval plus a bit
                time.sleep(config.update_interval_sec + 0.5)
                
                if cache_path.exists():
                    current_mtime = cache_path.stat().st_mtime
                    if current_mtime != last_mtime:
                        cache_updates += 1
                        last_mtime = current_mtime
                        
                        # Read cache content
                        with open(cache_path, 'r') as f:
                            cache_data = json.load(f)
                        
                        elapsed = time.time() - initial_time
                        print(f"Update #{cache_updates} at {elapsed:.1f}s: {len(cache_data.get('items', []))} cache items")
                
                print(f"Cycle {i+1}/3 completed")
            
            print(f"\nTest Results:")
            print(f"- Total cache updates: {cache_updates}")
            print(f"- Expected updates: ~3")
            print(f"- Cache file exists: {cache_path.exists()}")
            
            if cache_updates >= 2:
                print("✅ SUCCESS: Background timer is working!")
                print("   Dashboard will continue updating even when event queue is idle.")
                return True
            else:
                print("❌ FAILED: Background timer not updating cache as expected")
                return False
                
        finally:
            # Clean up
            glue.set_shutdown()
            time.sleep(0.5)  # Allow cleanup to complete


if __name__ == "__main__":
    success = test_background_timer()
    sys.exit(0 if success else 1)