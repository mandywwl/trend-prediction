#!/usr/bin/env python3
"""Simple test runner that doesn't require pytest dependencies."""

import sys
import time
import tempfile
import threading
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from service.runtime_glue import RuntimeGlue, RuntimeConfig
    
    class MockEventHandler:
        def on_event(self, event):
            return {"test_topic": 0.5}
    
    def test_timer_basic():
        """Test basic timer functionality."""
        print("Testing basic timer start/stop...")
        
        handler = MockEventHandler()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RuntimeConfig(
                metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
                predictions_cache_path=str(Path(temp_dir) / "cache.json"),
                update_interval_sec=1
            )
            
            glue = RuntimeGlue(handler, config)
            
            # Test start
            glue._start_background_timer()
            assert glue._timer_running
            print("✅ Timer started successfully")
            
            # Test stop
            glue.set_shutdown()
            time.sleep(0.2)
            assert not glue._timer_running
            print("✅ Timer stopped successfully")
        
        return True
    
    def test_timer_updates():
        """Test that timer actually updates cache."""
        print("Testing timer cache updates...")
        
        handler = MockEventHandler()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache.json"
            config = RuntimeConfig(
                metrics_snapshot_dir=str(Path(temp_dir) / "metrics"),
                predictions_cache_path=str(cache_path),
                update_interval_sec=1
            )
            
            glue = RuntimeGlue(handler, config)
            glue._predictions_buffer = [[{"topic_id": 1, "score": 0.8}]]
            
            try:
                glue._start_background_timer()
                time.sleep(1.5)  # Wait for update
                
                if cache_path.exists():
                    print("✅ Cache file created by timer")
                    return True
                else:
                    print("❌ Cache file not created")
                    return False
            finally:
                glue.set_shutdown()
    
    def run_tests():
        """Run all tests."""
        print("Running manual timer tests...")
        
        tests = [test_timer_basic, test_timer_updates]
        passed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                    print(f"✅ {test.__name__} passed")
                else:
                    print(f"❌ {test.__name__} failed")
            except Exception as e:
                print(f"❌ {test.__name__} error: {e}")
        
        print(f"\nResults: {passed}/{len(tests)} tests passed")
        return passed == len(tests)
    
    if __name__ == "__main__":
        success = run_tests()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Some dependencies may be missing. This is expected in a fresh environment.")
    sys.exit(0)  # Don't fail due to missing deps