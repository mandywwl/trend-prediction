#!/usr/bin/env python3
"""Integration test to verify real latency measurement works end-to-end."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_latency_integration():
    """Test that latency measurements are captured and stored in metrics files."""
    print("Testing latency integration...")
    
    try:
        # Import required modules
        from service.main import IntegratedEventHandler, periodic_preprocessing
        from service.runtime_glue import RuntimeGlue, RuntimeConfig
        from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
        from data_pipeline.storage.builder import GraphBuilder
        from model.inference.spam_filter import SpamScorer
        from model.inference.adaptive_thresholds import SensitivityController
        from config.schemas import Event
        from utils.io import get_hour_bucket
        
        print("✓ All modules imported successfully")
        
        # Initialize components (minimal setup for testing)
        spam_scorer = SpamScorer()
        graph_builder = GraphBuilder(spam_scorer=spam_scorer)
        sensitivity_controller = SensitivityController()
        
        # Use a mock embedder to avoid heavy dependencies in test
        class MockEmbedder:
            def __init__(self):
                self.batch_size = 1
                self.max_latency_ms = 10
                self.device = "cpu"
            def encode(self, texts):
                import numpy as np
                time.sleep(0.001)  # Simulate some processing time
                return [np.zeros(768) for _ in texts]
        
        embedder = MockEmbedder()
        
        # Create integrated event handler with latency measurement
        event_handler = IntegratedEventHandler(
            embedder=embedder,
            graph_builder=graph_builder,
            spam_scorer=spam_scorer,
            sensitivity_controller=sensitivity_controller
        )
        
        print("✓ Event handler created with latency aggregator")
        
        # Verify latency aggregator is initialized
        assert hasattr(event_handler, 'latency_aggregator'), "LatencyAggregator not initialized"
        print("✓ LatencyAggregator properly initialized")
        
        # Create test config and RuntimeGlue
        config = RuntimeConfig(
            update_interval_sec=1,  # Quick updates for testing
            enable_background_timer=False,  # Disable background timer for test
            metrics_snapshot_dir="data/metrics_hourly"
        )
        
        runtime_glue = RuntimeGlue(event_handler, config)
        
        # Process some test events to generate latency data
        test_events = [
            {
                'event_id': f'test_event_{i}',
                'ts_iso': datetime.now(timezone.utc).isoformat(),
                'actor_id': f'test_user_{i}',
                'text': f'Test content {i}',
                'features': {}
            }
            for i in range(5)
        ]
        
        print(f"Processing {len(test_events)} test events...")
        
        # Process events through the handler
        for event in test_events:
            scores = event_handler.on_event(event)
            print(f"Processed event {event['event_id']}: {len(scores)} scores")
            
            # Small delay to create measurable latencies
            time.sleep(0.001)
        
        # Check that latency measurements were collected
        measurements = event_handler.latency_aggregator.measurements
        print(f"✓ Collected {len(measurements)} latency measurements")
        assert len(measurements) > 0, "No latency measurements recorded"
        
        # Get latency summary
        latency_summary = event_handler.latency_aggregator.get_summary()
        print(f"Latency Summary: {latency_summary}")
        
        # Verify non-zero latencies
        assert latency_summary['median_ms'] > 0, f"Expected non-zero median latency, got {latency_summary['median_ms']}"
        assert latency_summary['p95_ms'] > 0, f"Expected non-zero p95 latency, got {latency_summary['p95_ms']}"
        
        print(f"✓ Non-zero latencies recorded:")
        print(f"  Median: {latency_summary['median_ms']}ms")
        print(f"  P95: {latency_summary['p95_ms']}ms")
        
        # Test RuntimeGlue integration
        runtime_glue._update_metrics_and_cache()
        
        # Check that metrics files were created with real latency data
        metrics_dir = Path("data/metrics_hourly")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the most recent metrics file
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        if metrics_files:
            latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
            print(f"✓ Found metrics file: {latest_file}")
            
            with open(latest_file) as f:
                data = json.load(f)
                latency_data = data.get('latency', {})
                
                print(f"Metrics file latency data: {latency_data}")
                
                # Verify the metrics contain real latency values
                if latency_data.get('median_ms', 0) > 0:
                    print(f"✓ Metrics file contains real latency data:")
                    print(f"  Median: {latency_data.get('median_ms')}ms")  
                    print(f"  P95: {latency_data.get('p95_ms')}ms")
                    
                    per_stage = latency_data.get('per_stage_ms', {})
                    if per_stage:
                        print(f"  Per-stage breakdown:")
                        for stage, ms in per_stage.items():
                            print(f"    {stage}: {ms}ms")
                    
                    return True
                else:
                    print("❌ Metrics file contains zero latency values")
                    return False
        else:
            print("❌ No metrics files found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_latency_integration()
    if success:
        print("\n✅ Latency integration test completed successfully!")
        print("Real latency measurements are now being captured and stored.")
    else:
        print("\n❌ Latency integration test failed!")
        sys.exit(1)