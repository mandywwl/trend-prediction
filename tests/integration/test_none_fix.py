"""Test script to verify None comparison fixes."""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model.evaluation.metrics import PrecisionAtKOnline


def test_none_timestamp_handling():
    """Test that None timestamps don't cause comparison errors."""
    print("Testing None timestamp handling in PrecisionAtKOnline...")
    
    # Initialize the online evaluator
    evaluator = PrecisionAtKOnline(
        delta_hours=2,
        window_min=60,
        k_default=5
    )
    
    # Test 1: Normal operation with valid timestamps
    print("Test 1: Valid timestamps")
    now = datetime.now(timezone.utc)
    ts_iso = now.isoformat()
    
    # Record some events and predictions
    evaluator.record_event(topic_id=1, user_id="user1", ts_iso=ts_iso)
    evaluator.record_predictions(ts_iso=ts_iso, items=[(1, 0.8), (2, 0.6)])
    
    # Get scores (should work without error)
    scores = evaluator.rolling_hourly_scores()
    print(f"✓ Valid timestamps work: {scores}")
    
    # Test 2: Empty timestamp string (should be handled gracefully)
    print("Test 2: Empty timestamp")
    try:
        evaluator.record_predictions(ts_iso="", items=[(3, 0.7)])
        print("✓ Empty timestamp handled gracefully")
    except Exception as e:
        print(f"✗ Empty timestamp caused error: {e}")
        return False
    
    # Test 3: None-like values in cache data (test runtime_glue fix)
    print("Test 3: Rolling scores with potential None timestamps")
    try:
        # This should not fail even if there are entries with None timestamps internally
        scores = evaluator.rolling_hourly_scores()
        print(f"✓ Rolling scores work: {scores}")
    except Exception as e:
        print(f"✗ Rolling scores failed: {e}")
        return False
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    success = test_none_timestamp_handling()
    sys.exit(0 if success else 1)
