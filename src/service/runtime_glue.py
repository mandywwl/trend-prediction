"""Runtime glue for streaming trend prediction service.

Provides a single entrypoint that:
- Runs the stream loop calling EventHandler.on_event
- Periodically resolves Δ-frozen predictions into Precision@K
- Writes hourly metrics and updates dashboard cache
- Supports YAML config overrides
- Handles graceful SIGINT shutdown
"""

import json
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Generator
from dataclasses import dataclass, asdict

from config.config import (
    DELTA_HOURS,
    WINDOW_MIN,
    K_DEFAULT,
    K_OPTIONS,
    METRICS_SNAPSHOT_DIR,
    PREDICTIONS_CACHE_PATH,
    TOPIC_LOOKUP_PATH,
)
from config.schemas import (
    Event,
    HourlyMetrics,
    PrecisionAtKSnapshot,
    PredictionsCache,
    CacheItem,
)
from model.evaluation.metrics import PrecisionAtKOnline
from utils.io import MetricsWriter, get_hour_bucket, ensure_dir, maybe_load_yaml


@dataclass
class RuntimeConfig:
    """Runtime configuration with YAML override support."""
    delta_hours: int = DELTA_HOURS
    window_min: int = WINDOW_MIN
    k_default: int = K_DEFAULT
    k_options: tuple = K_OPTIONS
    metrics_snapshot_dir: str = METRICS_SNAPSHOT_DIR
    predictions_cache_path: str = PREDICTIONS_CACHE_PATH
    topic_lookup_path: str = TOPIC_LOOKUP_PATH
    update_interval_sec: int = 60  # Update metrics every minute
    enable_background_timer: bool = True  # Enable background dashboard updates
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> 'RuntimeConfig':
        """Create config with optional YAML overrides."""
        yaml_config = maybe_load_yaml(yaml_path)
        
        # Extract runtime section if it exists
        runtime_config = yaml_config.get('runtime', {}) if isinstance(yaml_config, dict) else {}
        
        return cls(
            delta_hours=runtime_config.get('delta_hours', DELTA_HOURS),
            window_min=runtime_config.get('window_min', WINDOW_MIN),
            k_default=runtime_config.get('k_default', K_DEFAULT),
            k_options=tuple(runtime_config.get('k_options', K_OPTIONS)),
            metrics_snapshot_dir=runtime_config.get('metrics_snapshot_dir', METRICS_SNAPSHOT_DIR),
            predictions_cache_path=runtime_config.get('predictions_cache_path', PREDICTIONS_CACHE_PATH),
            topic_lookup_path=runtime_config.get('topic_lookup_path', TOPIC_LOOKUP_PATH),
            update_interval_sec=runtime_config.get('update_interval_sec', 60),
            enable_background_timer=runtime_config.get('enable_background_timer', True),
        )


class RuntimeGlue:
    """Main runtime component that orchestrates streaming and metrics."""
    
    def __init__(
        self, 
        event_handler: Any,  # Accepts any event handler that has on_event method
        config: Optional[RuntimeConfig] = None
    ):
        self.event_handler = event_handler
        self.config = config or RuntimeConfig()
        
        # Initialize metrics tracking
        self.precision_tracker = PrecisionAtKOnline(
            delta_hours=self.config.delta_hours,
            window_min=self.config.window_min,
            k_default=self.config.k_default,
            k_options=self.config.k_options,
        )
        
        # Initialize metrics writer with custom directory
        self.metrics_writer = MetricsWriter()
        self.metrics_writer.base_dir = Path(self.config.metrics_snapshot_dir)
        ensure_dir(self.metrics_writer.base_dir)
        
        # Ensure output directories exist
        ensure_dir(self.config.metrics_snapshot_dir)
        ensure_dir(Path(self.config.predictions_cache_path).parent)
        ensure_dir(Path(self.config.topic_lookup_path).parent)

        self._topic_lookup_path = Path(self.config.topic_lookup_path)
        if self._topic_lookup_path.exists():
            try:
                with open(self._topic_lookup_path, 'r', encoding='utf-8') as f:
                    self._topic_lookup = json.load(f)
            except Exception:
                self._topic_lookup = {}
        else:
            self._topic_lookup = {}
        
        # Runtime state
        self._running = False
        self._shutdown_event = threading.Event()
        self._last_update = datetime.now(timezone.utc)
        self._predictions_buffer = []  # Buffer recent predictions for cache
        
        # Background timer for dashboard updates
        self._timer_thread = None
        self._timer_running = False
        self._update_lock = threading.Lock()  # Prevent concurrent updates
    
    def set_shutdown(self):
        """Set shutdown event (called externally by main orchestrator)."""
        self._shutdown_event.set()
        self._running = False
        self._stop_background_timer()
    
    def _start_background_timer(self):
        """Start background timer for periodic dashboard updates."""
        if not self.config.enable_background_timer:
            print("Background timer disabled by configuration")
            return
        
        if self._timer_running or self._timer_thread is not None:
            return
        
        self._timer_running = True
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()
        print(f"Started background timer for dashboard updates (interval: {self.config.update_interval_sec}s)")
    
    def _stop_background_timer(self):
        """Stop background timer."""
        if not self._timer_running:
            return
        
        self._timer_running = False
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=2.0)
        self._timer_thread = None
        print("Stopped background timer")
    
    def _timer_loop(self):
        """Background timer loop for periodic dashboard updates."""
        while self._timer_running and not self._shutdown_event.is_set():
            try:
                # Sleep for the update interval, but check shutdown periodically
                for _ in range(self.config.update_interval_sec):
                    if not self._timer_running or self._shutdown_event.is_set():
                        break
                    time.sleep(1.0)
                
                # Update metrics and cache if still running and due for update
                if self._timer_running and not self._shutdown_event.is_set():
                    if self._should_update_metrics():
                        self._update_metrics_and_cache()
                    
            except Exception as e:
                print(f"Error in background timer: {e}")
                # Continue the loop despite errors
                time.sleep(5.0)  # Wait a bit before retrying
    
    def _should_update_metrics(self) -> bool:
        """Check if it's time to update metrics (every minute)."""
        now = datetime.now(timezone.utc)
        return (now - self._last_update).total_seconds() >= self.config.update_interval_sec
    
    def _update_metrics_and_cache(self):
        """Update rolling P@K, write hourly snapshots, and update predictions cache."""
        with self._update_lock:
            now = datetime.now(timezone.utc)

            # Get rolling precision scores
            precision_snapshot = self.precision_tracker.rolling_hourly_scores()
            adaptivity_score = self.precision_tracker.rolling_adaptivity_score()
            
            # Create hourly metrics (simplified - in real impl would include latency data)
            hourly_metrics = HourlyMetrics(
                precision_at_k=precision_snapshot,
                adaptivity=adaptivity_score,
                latency={
                    'median_ms': 0,  # Would be populated from LatencyAggregator
                    'p95_ms': 0,
                    'per_stage_ms': {
                        'ingest': 0,
                        'preprocess': 0,
                        'model_update_forward': 0,
                        'postprocess': 0
                    }
                },
                meta={
                    'service': 'runtime_glue',
                    'config_delta_hours': str(self.config.delta_hours),
                    'config_window_min': str(self.config.window_min)
                }
            )
            
            # Write hourly snapshot
            hour_bucket = get_hour_bucket(now)
            try:
                self.metrics_writer.write_hourly_snapshot(hour_bucket, hourly_metrics)
                print(f"[{now.isoformat()}] Wrote hourly metrics snapshot for {hour_bucket}")
            except Exception as e:
                print(f"[{now.isoformat()}] Error writing metrics snapshot: {e}")
            
            # Update predictions cache
            self._update_predictions_cache(now)
            
            # Update last update time
            self._last_update = now
    
    def _update_predictions_cache(self, now: datetime):
        """Update the predictions cache with recent predictions."""
        try:
            # Create cache item from recent predictions
            if self._predictions_buffer:
                # Use the most recent predictions
                latest_predictions = self._predictions_buffer[-1] if self._predictions_buffer else []
                
                cache_item = CacheItem(
                    t_iso=now.isoformat(),
                    topk=latest_predictions[:self.config.k_default]  # Limit to k_default
                )
                
                # Load existing cache or create new one
                cache_path = Path(self.config.predictions_cache_path)
                if cache_path.exists():
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                else:
                    cache_data = {"last_updated": "", "items": []}
                
                # Update cache
                cache_data["last_updated"] = now.isoformat()
                cache_data["items"].append(cache_item)
                
                # Keep only recent items (last hour)
                cutoff_time = now - timedelta(hours=1)
                cache_data["items"] = [
                    item for item in cache_data["items"]
                    if datetime.fromisoformat(item["t_iso"].replace('Z', '+00:00')) > cutoff_time
                ]
                
                # Write cache atomically
                temp_path = cache_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
                temp_path.replace(cache_path)
                
                print(f"[{now.isoformat()}] Updated predictions cache with {len(cache_data['items'])} items")

        except Exception as e:
            print(f"[{now.isoformat()}] Error updating predictions cache: {e}")

    def _update_topic_lookup(self, topic_id: int, label: str) -> None:
        """Persist mapping from topic_id to original label."""
        if not label:
            return
        try:
            key = str(topic_id)
            if self._topic_lookup.get(key) != label:
                self._topic_lookup[key] = label
                with open(self._topic_lookup_path, 'w', encoding='utf-8') as f:
                    json.dump(self._topic_lookup, f, indent=2)
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Error updating topic lookup: {e}")

    def _record_event_for_metrics(self, event: Event, scores: Dict[str, float]):
        """Record event and scores for metrics tracking."""
        try:
            # Extract topic_id and user_id from event
            topic_id = hash(event.get('event_id', '')) % 1000000  # Simple hash for demo
            self._update_topic_lookup(topic_id, event.get('event_id', ''))
            user_id = event.get('actor_id', 'unknown')
            ts_iso = event.get('ts_iso', datetime.now(timezone.utc).isoformat())
            
            # Record event for emergence labeling
            self.precision_tracker.record_event(
                topic_id=topic_id,
                user_id=user_id,
                ts_iso=ts_iso
            )
            
            # Record predictions if scores are available
            if scores:
                predictions = []
                for key, score in scores.items():
                    if isinstance(score, (int, float)):
                        # Convert string topic names to integer IDs for the metrics system
                        topic_id = abs(hash(key)) % 1000000
                        predictions.append((topic_id, score))
                        self._update_topic_lookup(topic_id, key)
                
                if predictions:
                    self.precision_tracker.record_predictions(
                        ts_iso=ts_iso,
                        items=predictions
                    )
                    
                    # Update predictions buffer for cache
                    formatted_predictions = [
                        {"topic_id": topic_id, "score": float(score)}
                        for topic_id, score in predictions
                    ]
                    self._predictions_buffer.append(formatted_predictions)
                    
                    # Keep buffer manageable
                    if len(self._predictions_buffer) > 100:
                        self._predictions_buffer = self._predictions_buffer[-50:]
        
        except Exception as e:
            # Never let metrics recording break the main loop
            print(f"Error recording metrics: {e}")
    
    def run_stream(self, event_stream: Iterable[Event]):
        """Run the main stream processing loop."""
        self._running = True
        print(f"Starting runtime glue with config: {asdict(self.config)}")
        
        # Start background timer for periodic dashboard updates
        self._start_background_timer()
        
        try:
            for event in event_stream:
                if not self._running or self._shutdown_event.is_set():
                    break
                
                # Process event through handler
                scores = self.event_handler.on_event(event)
                
                # Record for metrics tracking
                self._record_event_for_metrics(event, scores)
                
                # Check if it's time to update metrics (still check during event processing)
                if self._should_update_metrics():
                    self._update_metrics_and_cache()
                
                # Small delay to prevent tight loop
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """Perform graceful shutdown operations."""
        print("Performing graceful shutdown...")
        
        # Stop background timer first
        self._stop_background_timer()
        
        # Final metrics update
        try:
            self._update_metrics_and_cache()
            print("Final metrics update completed")
        except Exception as e:
            print(f"Error during final metrics update: {e}")
        
        print("Runtime glue shutdown complete")


# Mock stream generator for testing
def mock_event_stream(n_events: int = 100, delay: float = 0.1) -> Generator[Event, None, None]:
    """Generate mock events for testing."""
    for i in range(n_events):
        yield {
            'event_id': f'mock_event_{i}',
            'ts_iso': datetime.now(timezone.utc).isoformat(),
            'actor_id': f'user_{i % 10}',
            'target_ids': [f'target_{i % 5}'],
            'edge_type': 'mention',
            'features': {}
        }
        if delay > 0:
            time.sleep(delay)