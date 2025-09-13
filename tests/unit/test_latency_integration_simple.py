"""Simple unit test to verify latency measurement integration without heavy dependencies."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_latency_timer_and_aggregator():
    """Test the core latency measurement components directly."""
    print("Testing LatencyTimer and LatencyAggregator...")
    
    try:
        from utils.io import LatencyTimer, LatencyAggregator
        
        # Test LatencyTimer
        aggregator = LatencyAggregator()
        
        # Simulate multiple measurements
        for i in range(3):
            with LatencyTimer() as timer:
                timer.start_stage('ingest')
                time.sleep(0.001)  # 1ms
                timer.end_stage('ingest')
                
                timer.start_stage('preprocess')
                time.sleep(0.002)  # 2ms
                timer.end_stage('preprocess')
                
                timer.start_stage('model_update_forward')
                time.sleep(0.003)  # 3ms
                timer.end_stage('model_update_forward')
                
                timer.start_stage('postprocess')
                time.sleep(0.001)  # 1ms
                timer.end_stage('postprocess')
            
            # Record measurement
            aggregator.add_measurement(timer.total_duration_ms, timer.get_stage_ms())
            print(f"Measurement {i+1}: {timer.total_duration_ms}ms")
        
        # Get summary
        summary = aggregator.get_summary()
        print(f"Summary: {summary}")
        
        # Verify non-zero measurements
        assert summary['median_ms'] > 0, f"Expected non-zero median, got {summary['median_ms']}"
        assert summary['p95_ms'] > 0, f"Expected non-zero p95, got {summary['p95_ms']}"
        
        # Check per-stage measurements
        stages = summary['per_stage_ms']
        assert stages['ingest'] > 0, "Ingest stage should have non-zero latency"
        assert stages['preprocess'] > 0, "Preprocess stage should have non-zero latency"
        assert stages['model_update_forward'] > 0, "Model stage should have non-zero latency"
        assert stages['postprocess'] > 0, "Postprocess stage should have non-zero latency"
        
        print("✓ LatencyTimer and LatencyAggregator working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_event_handler_mock():
    """Test IntegratedEventHandler latency integration with minimal mocking."""
    print("\nTesting IntegratedEventHandler latency integration...")
    
    try:
        # Create minimal mock classes to avoid heavy dependencies
        class MockEmbedder:
            def __init__(self):
                self.batch_size = 1
                self.max_latency_ms = 10
                self.device = "cpu"
            def encode(self, texts):
                import numpy as np
                time.sleep(0.001)
                return [np.zeros(768) for _ in texts]
        
        class MockGraphBuilder:
            def __init__(self, spam_scorer=None):
                self.spam_scorer = spam_scorer
            def process_event(self, event):
                time.sleep(0.001)  # Simulate processing
        
        class MockSpamScorer:
            def edge_weight(self, event):
                return 1.0
        
        class MockSensitivityController:
            def policy(self):
                class Policy:
                    heavy_ops_enabled = True
                    sampler_size = 8
                return Policy()
            
            def record_event(self, is_spam, latency_ms):
                pass
        
        # Import the real IntegratedEventHandler
        from service.main import IntegratedEventHandler
        
        # Create handler with mocks
        spam_scorer = MockSpamScorer()
        graph_builder = MockGraphBuilder(spam_scorer)
        sensitivity_controller = MockSensitivityController()
        embedder = MockEmbedder()
        
        handler = IntegratedEventHandler(
            embedder=embedder,
            graph_builder=graph_builder,
            spam_scorer=spam_scorer,
            sensitivity_controller=sensitivity_controller
        )
        
        # Verify latency aggregator exists
        assert hasattr(handler, 'latency_aggregator'), "LatencyAggregator not initialized"
        print("✓ IntegratedEventHandler has latency_aggregator")
        
        # Create test event
        test_event = {
            'event_id': 'test_event_1',
            'ts_iso': datetime.now(timezone.utc).isoformat(),
            'actor_id': 'test_user',
            'text': 'Test content for latency measurement',
            'features': {}
        }
        
        # Process event (this should record latency)
        scores = handler.on_event(test_event)
        print(f"✓ Processed event, got {len(scores)} scores")
        
        # Check that latency was recorded
        measurements = handler.latency_aggregator.measurements
        assert len(measurements) > 0, "No latency measurements recorded"
        print(f"✓ Recorded {len(measurements)} latency measurements")
        
        # Get summary
        summary = handler.latency_aggregator.get_summary()
        print(f"Latency summary: {summary}")
        
        assert summary['median_ms'] > 0, f"Expected non-zero median latency, got {summary['median_ms']}"
        print("✓ Non-zero latency measurements recorded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_runtime_glue_integration():
    """Test RuntimeGlue latency integration."""
    print("\nTesting RuntimeGlue latency integration...")
    
    try:
        from service.runtime_glue import RuntimeGlue, RuntimeConfig
        from utils.io import LatencyAggregator
        
        # Create mock handler with latency aggregator
        class MockEventHandler:
            def __init__(self):
                self.latency_aggregator = LatencyAggregator()
                # Add some fake measurements
                for _ in range(3):
                    from utils.io import StageMs
                    stage_ms = StageMs(ingest=1, preprocess=2, model_update_forward=3, postprocess=1)
                    self.latency_aggregator.add_measurement(7, stage_ms)
            
            def on_event(self, event):
                return {'topic_123': 0.8}
        
        # Create RuntimeGlue with mock handler
        config = RuntimeConfig(
            metrics_snapshot_dir="data/test_metrics",
            enable_background_timer=False
        )
        
        runtime_glue = RuntimeGlue(MockEventHandler(), config)
        
        # Test _update_metrics_and_cache
        runtime_glue._update_metrics_and_cache()
        
        # Check that metrics files were created
        metrics_dir = Path("data/test_metrics")
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        
        if metrics_files:
            latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
            print(f"✓ Created metrics file: {latest_file}")
            
            with open(latest_file) as f:
                data = json.load(f)
                latency_data = data.get('latency', {})
                
                print(f"Latency data in metrics: {latency_data}")
                
                # Verify non-zero latency data
                assert latency_data.get('median_ms', 0) > 0, "Expected non-zero median in metrics"
                assert latency_data.get('p95_ms', 0) > 0, "Expected non-zero p95 in metrics"
                
                per_stage = latency_data.get('per_stage_ms', {})
                assert per_stage.get('ingest', 0) > 0, "Expected non-zero ingest latency"
                assert per_stage.get('preprocess', 0) > 0, "Expected non-zero preprocess latency"
                assert per_stage.get('model_update_forward', 0) > 0, "Expected non-zero model latency"
                assert per_stage.get('postprocess', 0) > 0, "Expected non-zero postprocess latency"
                
                print("✓ RuntimeGlue correctly recorded real latency data in metrics")
                return True
        else:
            print("❌ No metrics files created")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running latency integration tests...\n")
    
    test1_passed = test_latency_timer_and_aggregator()
    test2_passed = test_integrated_event_handler_mock()  
    test3_passed = test_runtime_glue_integration()
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✅ All latency integration tests passed!")
        print("Real latency measurement is successfully integrated!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)