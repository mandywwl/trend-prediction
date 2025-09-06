import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import numpy as np

from config.config import SLO_MED_MS, SLO_P95_MS, BUCKET_TZ, METRICS_SNAPSHOT_DIR
from config.schemas import LatencySummary, HourlyMetrics, StageMs


def maybe_load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config file with fallback to empty dict."""
    if not path:
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def ensure_dir(path: str | Path) -> Path:
    """Ensure the directory exists and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class LatencyTimer:
    """Context manager for timing operations with per-stage tracking."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.stage_times: Dict[str, float] = {}
        self.stage_durations: Dict[str, int] = {}
        self.total_duration_ms: Optional[int] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.perf_counter()
            self.total_duration_ms = int((end_time - self.start_time) * 1000)
    
    def start_stage(self, stage_name: str):
        """Start timing a specific stage."""
        self.stage_times[stage_name] = time.perf_counter()
    
    def end_stage(self, stage_name: str):
        """End timing a specific stage."""
        if stage_name in self.stage_times:
            duration_ms = int((time.perf_counter() - self.stage_times[stage_name]) * 1000)
            self.stage_durations[stage_name] = duration_ms
    
    def get_stage_ms(self) -> StageMs:
        """Get stage durations in the expected format."""
        return StageMs(
            ingest=self.stage_durations.get('ingest', 0),
            preprocess=self.stage_durations.get('preprocess', 0),
            model_update_forward=self.stage_durations.get('model_update_forward', 0),
            postprocess=self.stage_durations.get('postprocess', 0)
        )


class LatencyAggregator:
    """Aggregates latency measurements and compares against SLOs."""
    
    def __init__(self):
        self.measurements: List[int] = []
        self.stage_measurements: Dict[str, List[int]] = {
            'ingest': [],
            'preprocess': [],
            'model_update_forward': [],
            'postprocess': []
        }
    
    def add_measurement(self, total_ms: int, stage_ms: StageMs):
        """Add a latency measurement."""
        self.measurements.append(total_ms)
        self.stage_measurements['ingest'].append(stage_ms['ingest'])
        self.stage_measurements['preprocess'].append(stage_ms['preprocess'])
        self.stage_measurements['model_update_forward'].append(stage_ms['model_update_forward'])
        self.stage_measurements['postprocess'].append(stage_ms['postprocess'])
    
    def get_summary(self) -> LatencySummary:
        """Calculate latency summary with SLO comparison."""
        if not self.measurements:
            return LatencySummary(
                median_ms=0,
                p95_ms=0,
                per_stage_ms=StageMs(ingest=0, preprocess=0, model_update_forward=0, postprocess=0)
            )
        
        # Calculate percentiles
        median_ms = int(np.percentile(self.measurements, 50))
        p95_ms = int(np.percentile(self.measurements, 95))
        
        # Calculate per-stage means
        per_stage_ms = StageMs(
            ingest=int(np.mean(self.stage_measurements['ingest'])) if self.stage_measurements['ingest'] else 0,
            preprocess=int(np.mean(self.stage_measurements['preprocess'])) if self.stage_measurements['preprocess'] else 0,
            model_update_forward=int(np.mean(self.stage_measurements['model_update_forward'])) if self.stage_measurements['model_update_forward'] else 0,
            postprocess=int(np.mean(self.stage_measurements['postprocess'])) if self.stage_measurements['postprocess'] else 0
        )
        
        return LatencySummary(
            median_ms=median_ms,
            p95_ms=p95_ms,
            per_stage_ms=per_stage_ms
        )
    
    def meets_slo(self) -> Dict[str, bool]:
        """Check if current measurements meet SLO requirements."""
        summary = self.get_summary()
        return {
            'median_slo': summary['median_ms'] < SLO_MED_MS,
            'p95_slo': summary['p95_ms'] < SLO_P95_MS
        }
    
    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()
        for stage_list in self.stage_measurements.values():
            stage_list.clear()


class MetricsWriter:
    """Handles writing hourly metrics snapshots."""
    
    def __init__(self):
        self.base_dir = Path(METRICS_SNAPSHOT_DIR)
        ensure_dir(self.base_dir)
    
    def write_hourly_snapshot(self, ts_hour: datetime, payload: HourlyMetrics):
        """
        Write hourly metrics snapshot with idempotent upserts.
        
        Args:
            ts_hour: Datetime representing the hour bucket (should be timezone-aware)
            payload: HourlyMetrics data to write
        """
        # Ensure timezone awareness
        if ts_hour.tzinfo is None:
            # Assume UTC if no timezone info
            ts_hour = ts_hour.replace(tzinfo=timezone.utc)
        
        # Convert to bucket timezone if different
        if BUCKET_TZ != "UTC":
            import zoneinfo
            target_tz = zoneinfo.ZoneInfo(BUCKET_TZ)
            ts_hour = ts_hour.astimezone(target_tz)
        
        # Create filename based on hour bucket
        filename = f"metrics_{ts_hour.strftime('%Y%m%d_%H')}.json"
        filepath = self.base_dir / filename
        
        # Add generation timestamp to meta
        if 'meta' not in payload:
            payload['meta'] = {}
        payload['meta']['generated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Write with atomic operation (write to temp file then rename)
        temp_filepath = filepath.with_suffix('.tmp')
        with open(temp_filepath, 'w') as f:
            json.dump(payload, f, indent=2)
        
        # Atomic rename for idempotent upserts
        temp_filepath.replace(filepath)
    
    def read_hourly_snapshot(self, ts_hour: datetime) -> Optional[HourlyMetrics]:
        """Read an existing hourly snapshot if it exists."""
        # Ensure timezone awareness
        if ts_hour.tzinfo is None:
            ts_hour = ts_hour.replace(tzinfo=timezone.utc)
        
        # Convert to bucket timezone if different
        if BUCKET_TZ != "UTC":
            import zoneinfo
            target_tz = zoneinfo.ZoneInfo(BUCKET_TZ)
            ts_hour = ts_hour.astimezone(target_tz)
        
        filename = f"metrics_{ts_hour.strftime('%Y%m%d_%H')}.json"
        filepath = self.base_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None


def get_hour_bucket(dt: datetime) -> datetime:
    """
    Get the hour bucket for a given datetime in the configured timezone.
    
    Args:
        dt: Input datetime (timezone-aware)
    
    Returns:
        Datetime representing the start of the hour bucket
    """
    # Ensure timezone awareness
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to bucket timezone
    if BUCKET_TZ != "UTC":
        import zoneinfo
        target_tz = zoneinfo.ZoneInfo(BUCKET_TZ)
        dt = dt.astimezone(target_tz)
    
    # Round down to the hour
    return dt.replace(minute=0, second=0, microsecond=0)
