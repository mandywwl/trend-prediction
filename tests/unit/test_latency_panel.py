"""Tests for latency dashboard panel functionality."""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add paths for imports
dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(dashboard_path))
sys.path.insert(0, str(src_path))

from dashboard.components.latency import (
    _load_hourly_metrics, 
    _safe_parse_latency,
    _filter_slo_breaches
)
from config.config import SLO_MED_MS, SLO_P95_MS


def test_safe_parse_latency():
    """Test safe parsing of latency data from metrics."""
    # Valid data
    valid_data = {
        "latency": {
            "median_ms": 500,
            "p95_ms": 1200,
            "per_stage_ms": {
                "ingest": 50,
                "preprocess": 100,
                "model_update_forward": 200,
                "postprocess": 150
            }
        }
    }
    
    latency = _safe_parse_latency(valid_data)
    assert latency is not None
    assert latency['median_ms'] == 500
    assert latency['p95_ms'] == 1200
    assert latency['per_stage_ms']['ingest'] == 50
    assert latency['per_stage_ms']['preprocess'] == 100
    assert latency['per_stage_ms']['model_update_forward'] == 200
    assert latency['per_stage_ms']['postprocess'] == 150
    
    # Missing data
    invalid_data = {"other": "data"}
    latency = _safe_parse_latency(invalid_data)
    assert latency is None
    
    # Partial data (should still work with defaults)
    partial_data = {
        "latency": {
            "median_ms": 800,
            "p95_ms": 1500
            # missing per_stage_ms
        }
    }
    latency = _safe_parse_latency(partial_data)
    assert latency is not None
    assert latency['median_ms'] == 800
    assert latency['p95_ms'] == 1500
    assert latency['per_stage_ms']['ingest'] == 0  # default


def test_filter_slo_breaches():
    """Test filtering of SLO breaches."""
    # Create test data with some breaches
    now = datetime.now(timezone.utc)
    test_data = [
        (now - timedelta(hours=3), {
            "latency": {
                "median_ms": 500,  # Under SLO
                "p95_ms": 800,     # Under SLO
                "per_stage_ms": {"ingest": 50, "preprocess": 100, "model_update_forward": 200, "postprocess": 150}
            }
        }),
        (now - timedelta(hours=2), {
            "latency": {
                "median_ms": 1200,  # Over median SLO (1000)
                "p95_ms": 1800,     # Under P95 SLO (2000)
                "per_stage_ms": {"ingest": 60, "preprocess": 110, "model_update_forward": 210, "postprocess": 820}
            }
        }),
        (now - timedelta(hours=1), {
            "latency": {
                "median_ms": 800,   # Under SLO
                "p95_ms": 2500,     # Over P95 SLO (2000)
                "per_stage_ms": {"ingest": 70, "preprocess": 120, "model_update_forward": 220, "postprocess": 2090}
            }
        })
    ]
    
    # Test without filtering (should return all valid entries)
    all_data = _filter_slo_breaches(test_data, show_breaches_only=False)
    assert len(all_data) == 3
    
    # Test with breach filtering (should return only breached entries)
    breach_data = _filter_slo_breaches(test_data, show_breaches_only=True)
    assert len(breach_data) == 2  # 2nd and 3rd entries have breaches
    
    # Check that breach filtering correctly identifies breaches
    assert breach_data[0][1]['median_ms'] == 1200  # Median breach
    assert breach_data[1][1]['p95_ms'] == 2500     # P95 breach


def test_load_hourly_metrics():
    """Test loading hourly metrics from directory."""
    # Create temporary directory with test metrics files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test metrics files
        test_files = [
            ("metrics_20240115_10.json", {
                "latency": {
                    "median_ms": 500,
                    "p95_ms": 1200,
                    "per_stage_ms": {"ingest": 50, "preprocess": 100, "model_update_forward": 200, "postprocess": 150}
                },
                "meta": {"generated_at": "2024-01-15T10:30:00Z"}
            }),
            ("metrics_20240115_11.json", {
                "latency": {
                    "median_ms": 600,
                    "p95_ms": 1300,
                    "per_stage_ms": {"ingest": 60, "preprocess": 110, "model_update_forward": 210, "postprocess": 220}
                },
                "meta": {"generated_at": "2024-01-15T11:30:00Z"}
            }),
            ("invalid_file.json", {"invalid": "data"}),  # Should be skipped
            ("not_json.txt", "not json")  # Should be skipped
        ]
        
        for filename, data in test_files:
            file_path = temp_path / filename
            if filename.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(data, f)
            else:
                with open(file_path, 'w') as f:
                    f.write(data)
        
        # Load metrics
        metrics_data = _load_hourly_metrics(str(temp_path))
        
        # Should load 2 valid files, skip invalid ones
        assert len(metrics_data) == 2
        
        # Check sorting by timestamp
        assert metrics_data[0][0] < metrics_data[1][0]
        
        # Check data content
        assert metrics_data[0][1]['latency']['median_ms'] == 500
        assert metrics_data[1][1]['latency']['median_ms'] == 600


def test_load_hourly_metrics_empty_dir():
    """Test loading from empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_data = _load_hourly_metrics(temp_dir)
        assert len(metrics_data) == 0


def test_load_hourly_metrics_nonexistent_dir():
    """Test loading from non-existent directory."""
    metrics_data = _load_hourly_metrics("/path/that/does/not/exist")
    assert len(metrics_data) == 0


def create_test_metrics_data():
    """Create sample metrics data for manual testing."""
    # Create test directory
    test_dir = Path("/tmp/test_metrics")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample data for the last 24 hours
    now = datetime.now(timezone.utc)
    
    for i in range(24):
        timestamp = now - timedelta(hours=i)
        filename = f"metrics_{timestamp.strftime('%Y%m%d_%H')}.json"
        
        # Simulate some data patterns
        base_median = 800 + (i * 20)  # Increasing trend
        base_p95 = 1500 + (i * 50)   # Increasing trend
        
        # Add some random variation and occasional breaches
        if i % 6 == 0:  # Every 6 hours, create a breach
            median_ms = SLO_MED_MS + 200  # Breach median SLO
            p95_ms = base_p95
        elif i % 8 == 0:  # Different pattern for P95 breaches
            median_ms = base_median
            p95_ms = SLO_P95_MS + 300  # Breach P95 SLO
        else:
            median_ms = base_median
            p95_ms = base_p95
        
        data = {
            "latency": {
                "median_ms": median_ms,
                "p95_ms": p95_ms,
                "per_stage_ms": {
                    "ingest": max(50, median_ms // 10),
                    "preprocess": max(100, median_ms // 8),
                    "model_update_forward": max(200, median_ms // 4),
                    "postprocess": max(150, median_ms // 6)
                }
            },
            "precision_at_k": {
                "k5": 0.85,
                "k10": 0.75,
                "support": 100
            },
            "meta": {
                "generated_at": timestamp.isoformat()
            }
        }
        
        with open(test_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Created test metrics data in {test_dir}")
    return str(test_dir)


if __name__ == "__main__":
    # Run tests
    test_safe_parse_latency()
    test_filter_slo_breaches()
    test_load_hourly_metrics()
    test_load_hourly_metrics_empty_dir()
    test_load_hourly_metrics_nonexistent_dir()
    
    print("All latency panel tests passed!")
    
    # Create sample data for manual testing
    print("\nCreating sample test data...")
    test_dir = create_test_metrics_data()
    print(f"Test data created in: {test_dir}")
    print(f"You can test the dashboard with: METRICS_SNAPSHOT_DIR={test_dir}")