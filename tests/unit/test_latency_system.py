"""Test script to validate latency tracking and SLO monitoring implementation."""

import json
import random
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.utils.io import LatencyTimer, LatencyAggregator, MetricsWriter, get_hour_bucket
from src.config.schemas import HourlyMetrics, PrecisionAtKSnapshot
from src.config.config import SLO_MED_MS, SLO_P95_MS, METRICS_SNAPSHOT_DIR


def test_latency_timer():
    """Test LatencyTimer basic functionality."""
    print("Testing LatencyTimer...")
    
    with LatencyTimer() as timer:
        timer.start_stage('ingest')
        time.sleep(0.01)  # 10ms
        timer.end_stage('ingest')
        
        timer.start_stage('preprocess')
        time.sleep(0.02)  # 20ms
        timer.end_stage('preprocess')
        
        timer.start_stage('model_update_forward')
        time.sleep(0.015)  # 15ms
        timer.end_stage('model_update_forward')
        
        timer.start_stage('postprocess')
        time.sleep(0.005)  # 5ms
        timer.end_stage('postprocess')
    
    stage_ms = timer.get_stage_ms()
    total_ms = timer.total_duration_ms
    
    print(f"Total duration: {total_ms}ms")
    print(f"Stage durations: {stage_ms}")
    
    # Check that stage times sum approximately to total
    stage_sum = sum(stage_ms.values())
    print(f"Stage sum: {stage_sum}ms, Total: {total_ms}ms, Diff: {abs(stage_sum - total_ms)}ms")
    
    assert 45 <= total_ms <= 65, f"Expected ~50ms total, got {total_ms}ms"
    assert stage_sum <= total_ms, "Stage sum should not exceed total time"
    print("✓ LatencyTimer test passed\n")


def test_synthetic_500_events():
    """Test with 500 synthetic events to validate hourly aggregation."""
    print("Testing with 500 synthetic events...")
    
    aggregator = LatencyAggregator()
    
    # Generate 500 synthetic latency measurements
    for i in range(500):
        # Simulate realistic latencies with some variation
        base_latency = random.uniform(500, 1500)  # 500-1500ms base
        
        # Add some outliers to test p95
        if i % 50 == 0:  # 2% outliers
            base_latency += random.uniform(1000, 2000)
        
        total_ms = int(base_latency)
        
        # Distribute latency across stages (with realistic proportions)
        ingest_ms = int(total_ms * random.uniform(0.1, 0.2))
        preprocess_ms = int(total_ms * random.uniform(0.3, 0.5))
        model_ms = int(total_ms * random.uniform(0.2, 0.4))
        postprocess_ms = max(1, total_ms - ingest_ms - preprocess_ms - model_ms)
        
        stage_ms = {
            'ingest': ingest_ms,
            'preprocess': preprocess_ms,
            'model_update_forward': model_ms,
            'postprocess': postprocess_ms
        }
        
        aggregator.add_measurement(total_ms, stage_ms)
    
    summary = aggregator.get_summary()
    slo_status = aggregator.meets_slo()
    
    print(f"Median latency: {summary['median_ms']}ms (SLO: {SLO_MED_MS}ms)")
    print(f"P95 latency: {summary['p95_ms']}ms (SLO: {SLO_P95_MS}ms)")
    print(f"Per-stage means: {summary['per_stage_ms']}")
    print(f"SLO compliance: {slo_status}")
    
    # Validate that stage means sum approximately to total median
    stage_sum = sum(summary['per_stage_ms'].values())
    print(f"Stage means sum: {stage_sum}ms, Median: {summary['median_ms']}ms")
    
    # Check sensible values
    assert 100 <= summary['median_ms'] <= 3000, f"Median should be sensible, got {summary['median_ms']}ms"
    assert summary['p95_ms'] >= summary['median_ms'], "P95 should be >= median"
    assert stage_sum > 0, "Stage sum should be positive"
    
    print("✓ Synthetic 500 events test passed\n")
    return summary


def test_metrics_writer():
    """Test MetricsWriter with idempotent upserts."""
    print("Testing MetricsWriter...")
    
    writer = MetricsWriter()
    
    # Create test hour bucket
    test_time = datetime.now(timezone.utc)
    hour_bucket = get_hour_bucket(test_time)
    
    # Create test metrics payload
    test_metrics = HourlyMetrics(
        precision_at_k=PrecisionAtKSnapshot(k5=0.85, k10=0.75, support=100),
        latency={
            'median_ms': 800,
            'p95_ms': 1200,
            'per_stage_ms': {
                'ingest': 80,
                'preprocess': 300,
                'model_update_forward': 350,
                'postprocess': 70
            }
        },
        meta={"test": "data"}
    )
    
    # Write initial snapshot
    writer.write_hourly_snapshot(hour_bucket, test_metrics)
    
    # Read it back
    read_metrics = writer.read_hourly_snapshot(hour_bucket)
    assert read_metrics is not None, "Should be able to read back written metrics"
    assert read_metrics['latency']['median_ms'] == 800, "Data should match"
    assert 'generated_at' in read_metrics['meta'], "Should have generation timestamp"
    
    # Test idempotent upsert (write again with different data)
    test_metrics['latency']['median_ms'] = 750
    writer.write_hourly_snapshot(hour_bucket, test_metrics)
    
    # Read again
    updated_metrics = writer.read_hourly_snapshot(hour_bucket)
    assert updated_metrics['latency']['median_ms'] == 750, "Should have updated value"
    
    print(f"✓ Successfully wrote and read metrics for hour: {hour_bucket}")
    print("✓ MetricsWriter test passed\n")


def test_hour_bucketing():
    """Test hour bucketing in configured timezone."""
    print("Testing hour bucketing...")
    
    # Test various times
    test_times = [
        datetime(2025, 9, 5, 14, 30, 45, tzinfo=timezone.utc),
        datetime(2025, 9, 5, 23, 59, 59, tzinfo=timezone.utc),
        datetime(2025, 9, 6, 0, 0, 1, tzinfo=timezone.utc),
    ]
    
    for dt in test_times:
        bucket = get_hour_bucket(dt)
        print(f"Input: {dt} -> Bucket: {bucket}")
        
        # Verify bucket is start of hour
        assert bucket.minute == 0, "Bucket should be start of hour"
        assert bucket.second == 0, "Bucket should be start of hour"
        assert bucket.microsecond == 0, "Bucket should be start of hour"
        
        # Verify it's in the same hour or earlier
        assert bucket <= dt, "Bucket should be <= input time"
        assert (dt - bucket).total_seconds() < 3600, "Should be within same hour"
    
    print("✓ Hour bucketing test passed\n")


def test_end_to_end():
    """End-to-end test combining all components."""
    print("Testing end-to-end workflow...")
    
    aggregator = LatencyAggregator()
    writer = MetricsWriter()
    
    # Simulate some events with timing
    for i in range(10):
        with LatencyTimer() as timer:
            timer.start_stage('ingest')
            time.sleep(0.001 * random.uniform(0.5, 2.0))  # 0.5-2ms
            timer.end_stage('ingest')
            
            timer.start_stage('preprocess')
            time.sleep(0.001 * random.uniform(2, 5))  # 2-5ms
            timer.end_stage('preprocess')
            
            timer.start_stage('model_update_forward')
            time.sleep(0.001 * random.uniform(3, 6))  # 3-6ms
            timer.end_stage('model_update_forward')
            
            timer.start_stage('postprocess')
            time.sleep(0.001 * random.uniform(0.5, 1.5))  # 0.5-1.5ms
            timer.end_stage('postprocess')
        
        aggregator.add_measurement(timer.total_duration_ms, timer.get_stage_ms())
    
    # Get summary and create hourly metrics
    latency_summary = aggregator.get_summary()
    
    hourly_metrics = HourlyMetrics(
        precision_at_k=PrecisionAtKSnapshot(k5=0.9, k10=0.8, support=50),
        latency=latency_summary,
        meta={"source": "end_to_end_test"}
    )
    
    # Write to snapshot
    current_hour = get_hour_bucket(datetime.now(timezone.utc))
    writer.write_hourly_snapshot(current_hour, hourly_metrics)
    
    print(f"✓ End-to-end test completed successfully")
    print(f"  Final latency summary: {latency_summary}")
    print(f"  Written to hour bucket: {current_hour}")


def main():
    """Run all tests."""
    print("=== Latency Tracking and SLO Monitoring Tests ===\n")
    
    test_latency_timer()
    test_synthetic_500_events()
    test_metrics_writer() 
    test_hour_bucketing()
    test_end_to_end()
    
    print("=== All tests passed! ===")
    print(f"\nMetrics are being written to: {METRICS_SNAPSHOT_DIR}")
    print(f"SLO thresholds: Median < {SLO_MED_MS}ms, P95 < {SLO_P95_MS}ms")


if __name__ == "__main__":
    main()
