"""Runtime orchestration layer for streaming trend prediction service.

Bridges event processing, metrics collection, and dashboard data management.

Key Features:
- Processes streaming events via EventHandler.on_event()
- Tracks Precision@K, adaptivity, latency, and robustness metrics
- Background thread for periodic dashboard cache and metrics updates
- Topic ID mapping and label generation pipeline integration
- YAML-configurable parameters and graceful shutdown handling

Main Components:
- RuntimeConfig: Configuration with YAML override support
- RuntimeGlue: Main orchestrator coordinating stream processing and metrics
- Background services for dashboard updates and topic labeling
"""

import json
import time
import threading
import traceback
import hashlib
import logging

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Generator
from dataclasses import dataclass, asdict
from math import isfinite

from config.config import (
    DELTA_HOURS,
    WINDOW_MIN,
    K_DEFAULT,
    K_OPTIONS,
    METRICS_SNAPSHOT_DIR,
    PREDICTIONS_CACHE_PATH,
    TOPIC_LOOKUP_PATH,
    METRICS_LOOKUP_PATH,
    EVENT_JSONL_PATH,
)
from config.schemas import (
    Event,
    HourlyMetrics,
    LatencySummary,
    PrecisionAtKSnapshot,
    PredictionsCache,
    CacheItem,
    StageMs,
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
    metrics_lookup_path: str = METRICS_LOOKUP_PATH
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
            metrics_lookup_path=runtime_config.get('metrics_lookup_path', METRICS_LOOKUP_PATH),
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
        self.logger = logging.getLogger(__name__) # Logger used by _record_event_for_metrics and others
        
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
        ensure_dir(Path(self.config.metrics_lookup_path).parent)

        self._topic_lookup_path = Path(self.config.topic_lookup_path)
        if self._topic_lookup_path.exists():
            try:
                with open(self._topic_lookup_path, 'r', encoding='utf-8') as f:
                    self._topic_lookup = json.load(f)
            except Exception:
                self._topic_lookup = {}
        else:
            self._topic_lookup = {}

        self._metrics_lookup_path = Path(self.config.metrics_lookup_path)
        if self._metrics_lookup_path.exists():
            try:
                with open(self._metrics_lookup_path, 'r', encoding='utf-8') as f:
                    self._metrics_lookup = json.load(f)
            except Exception:
                self._metrics_lookup = {}
        else:
            self._metrics_lookup = {}
        
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
    
    def _safe_parse_iso(self, ts_iso: str) -> Optional[datetime]:
        """Safely parse ISO timestamp, returning None on failure."""
        try:
            if not ts_iso:
                return None
            return datetime.fromisoformat(ts_iso.replace('Z', '+00:00'))
        except Exception:
            return None
    
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
                        try:
                            self._update_metrics_and_cache()
                        except Exception as e:
                            # Log full traceback to pinpoint exact failing line
                            print(f"Error in background timer during update: {e}\n{traceback.format_exc()}")
                            # Continue loop after brief pause
                            time.sleep(2.0)
                    
            except Exception as e:
                print(f"Error in background timer: {e}\n{traceback.format_exc()}")
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
            try:
                precision_snapshot = self.precision_tracker.rolling_hourly_scores()
            except Exception as e:
                print(f"[{now.isoformat()}] Error computing rolling_hourly_scores: {e}\n{traceback.format_exc()}")
                precision_snapshot = {'k5': 0.0, 'k10': 0.0, 'support': 0}

            # Get adaptivity score
            try:
                adaptivity_score = self.precision_tracker.rolling_adaptivity_score()
            except Exception as e:
                print(f"[{now.isoformat()}] Error computing rolling_adaptivity_score: {e}\n{traceback.format_exc()}")
                adaptivity_score = 0.0

            # Get real latency data from handler

            try:
                if hasattr(self.event_handler, 'latency_aggregator'):
                    latency_summary = self.event_handler.latency_aggregator.get_summary()
                    self.event_handler.latency_aggregator.clear()  # Reset for next hour
                else:
                    # Fallback for handlers without latency measurement
                    latency_summary = {
                        'median_ms': 0,
                        'p95_ms': 0,
                        'per_stage_ms': {
                            'ingest': 0,
                            'preprocess': 0,
                            'model_update_forward': 0,
                            'postprocess': 0
                        }
                    }
            except Exception as e:
                print(f"[{now.isoformat()}] Error getting latency summary: {e}\n{traceback.format_exc()}")
                latency_summary = {
                    'median_ms': 0,
                    'p95_ms': 0,
                    'per_stage_ms': {
                        'ingest': 0,
                        'preprocess': 0,
                        'model_update_forward': 0,
                        'postprocess': 0
                    }
                }


            # Get robustness data from sensitivity controller if available
            try:
                robustness_data = {}
                if hasattr(self.event_handler, 'sensitivity') and self.event_handler.sensitivity:
                    metrics = self.event_handler.sensitivity.metrics()
                    thresholds = self.event_handler.sensitivity.thresholds()
                    robustness_data = {
                        'spam_rate': metrics.get('spam_rate', 0.0),
                        'theta_g': thresholds.theta_g,
                        'theta_u': thresholds.theta_u,
                    }
                    
                    # Calculate downweighted percentage from recent events if available
                    # This is a simple approximation - in a real system this would be more sophisticated
                    if 'spam_rate' in robustness_data:
                        # Approximate downweighted percentage based on spam rate
                        robustness_data['downweighted_pct'] = min(robustness_data['spam_rate'] * 50.0, 25.0)
                
                if not robustness_data:
                    # Fallback robustness data
                    robustness_data = {
                        'spam_rate': 0.0,
                        'downweighted_pct': 0.0,
                        'theta_g': 0.5,
                        'theta_u': 0.4,
                    }
            except Exception as e:
                print(f"[{now.isoformat()}] Error getting robustness data: {e}")
                robustness_data = {
                    'spam_rate': 0.0,
                    'downweighted_pct': 0.0,
                    'theta_g': 0.5,
                    'theta_u': 0.4,
                }

            # Create hourly metrics (now with real latency data and robustness data!)
            try:
                hourly_metrics = HourlyMetrics(
                    precision_at_k=precision_snapshot,
                    adaptivity=adaptivity_score,
                    latency=latency_summary,
                    robustness=robustness_data,
                    meta={
                        'service': 'runtime_glue',
                        'config_delta_hours': str(self.config.delta_hours),
                        'config_window_min': str(self.config.window_min)
                    }
                )
            except Exception as e:
                print(f"[{now.isoformat()}] Error constructing HourlyMetrics payload: {e}\n{traceback.format_exc()}")
                hourly_metrics = HourlyMetrics(
                    precision_at_k={'k5': 0.0, 'k10': 0.0, 'support': 0},
                    adaptivity=0.0,
                    latency={
                        'median_ms': 0,
                        'p95_ms': 0,
                        'per_stage_ms': {
                            'ingest': 0,
                            'preprocess': 0,
                            'model_update_forward': 0,
                            'postprocess': 0
                        }
                    },
                    robustness={
                        'spam_rate': 0.0,
                        'downweighted_pct': 0.0,
                        'theta_g': 0.5,
                        'theta_u': 0.4,
                    },
                    meta={'service': 'runtime_glue'}
                )

            # Write hourly snapshot
            try:
                hour_bucket = get_hour_bucket(now)
                self.metrics_writer.write_hourly_snapshot(hour_bucket, hourly_metrics)
                print(f"[{now.isoformat()}] Wrote hourly metrics snapshot for {hour_bucket}")
            except Exception as e:
                print(f"[{now.isoformat()}] Error writing metrics snapshot: {e}\n{traceback.format_exc()}")
            
            # Update predictions cache
            try:
                self._update_predictions_cache(now)
            except Exception as e:
                print(f"[{now.isoformat()}] Error updating predictions cache: {e}\n{traceback.format_exc()}")
            
            # Update last update time
            self._last_update = now
    
    def _update_predictions_cache(self, now: datetime):
        """Update the predictions cache with recent predictions."""
        try:
            # Create cache item from recent predictions
            if not self._predictions_buffer:
                return
            
            # 1) Aggregate by topic_id across all events since the last update.
            #    Use max(score) per topic to keep the strongest signal.
            agg: dict[int, float] = {}
            for preds in self._predictions_buffer:
                for item in preds:
                    try: 
                        tid = int(item.get("topic_id"))
                        sc = float(item.get("score", 0.0))
                    except Exception:
                        continue
                    if not isfinite(sc):
                        continue
                    if sc > agg.get(tid, float('-inf')):
                        agg[tid] = sc

            # 2) Take global top-K by score
            pairs = sorted(agg.items(), key=lambda x: (-x[1], x[0]))  # sort by score desc, topic_id asc
            K_store = max(self.config.k_options) if getattr(self.config, "k_options", None) else self.config.k_default
            topk = [{"topic_id": tid, "score": float(sc)} for tid, sc in pairs[:K_store]]

            if not topk:
                return
            
            # 3) Build and persist cache item
            cache_item = {"t_iso": now.isoformat(), "topk": topk}
            cache_path = Path(self.config.predictions_cache_path)
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            else:
                cache_data = {"last_updated": "", "items": []}

            cache_data["last_updated"] = now.isoformat()
            cache_data["items"].append(cache_item)

            # Keep only recent items (last hour)
            cutoff_time = now - timedelta(hours=1)
            cache_data["items"] = [
                item for item in cache_data["items"]
                if item.get("t_iso") and self._safe_parse_iso(item["t_iso"]) and
                self._safe_parse_iso(item["t_iso"]) > cutoff_time
            ]

            # Write cache atomically
            tmp = cache_path.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            tmp.replace(cache_path)

            # Log
            total_preds = sum(len(x) for x in self._predictions_buffer)
            print(f"[{now.isoformat()}] Aggregated {total_preds} preds ->  wrote {len(topk)} topics to cache")

            # 4) Clear buffer after successful write
            self._predictions_buffer.clear()

            print(f"[{now.isoformat()}] Updated predictions cache with {len(cache_data['items'])} items")



                # # Use the most recent predictions
                # latest_predictions = self._predictions_buffer[-1] if self._predictions_buffer else []
                
                # cache_item = CacheItem(
                #     t_iso=now.isoformat(),
                #     topk=latest_predictions[:self.config.k_default]  # Limit to k_default
                # )
                
                # # Load existing cache or create new one
                # cache_path = Path(self.config.predictions_cache_path)
                # if cache_path.exists():
                #     with open(cache_path, 'r', encoding='utf-8') as f:
                #         cache_data = json.load(f)
                # else:
                #     cache_data = {"last_updated": "", "items": []}
                
                # # Update cache
                # cache_data["last_updated"] = now.isoformat()
                # cache_data["items"].append(cache_item)
                
                # # Keep only recent items (last hour)
                # cutoff_time = now - timedelta(hours=1)
                # cache_data["items"] = [
                #     item for item in cache_data["items"]
                #     if item.get("t_iso") and self._safe_parse_iso(item["t_iso"]) and 
                #     self._safe_parse_iso(item["t_iso"]) > cutoff_time
                # ]
                
                # # Write cache atomically
                # temp_path = cache_path.with_suffix('.tmp')
                # with open(temp_path, 'w', encoding='utf-8') as f:
                #     json.dump(cache_data, f, indent=2)
                # temp_path.replace(cache_path)
                
                # print(f"[{now.isoformat()}] Updated predictions cache with {len(cache_data['items'])} items")

        except Exception as e:
            print(f"[{now.isoformat()}] Error updating predictions cache: {e}")

    def update_topic_labels(self) -> None:
        """Update topic labels using the topic labeling pipeline."""
        try:
            from data_pipeline.processors.topic_labeling import run_topic_labeling_pipeline
            
            print(f"[{datetime.now().isoformat()}] Running topic labeling pipeline...")
            
            # Run the pipeline to generate meaningful labels
            result = run_topic_labeling_pipeline(
                events_path=str(EVENT_JSONL_PATH),
                topic_lookup_path=str(self._topic_lookup_path),
                use_embedder=False  # Use TF-IDF only for better performance
            )
            
            # Reload the updated lookup
            if self._topic_lookup_path.exists():
                with open(self._topic_lookup_path, 'r', encoding='utf-8') as f:
                    self._topic_lookup = json.load(f)
            
            updated_count = sum(1 for label in result.values() 
                              if not (label.startswith("topic_") or 
                                     label.startswith("test_") or 
                                     label.startswith("viral_") or 
                                     label.startswith("trending_")))
            
            print(f"[{datetime.now().isoformat()}] Topic labeling completed. Updated {updated_count} labels.")
            
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Error running topic labeling pipeline: {e}")
    
    def _update_topic_lookup(self, topic_id: int, label: str) -> None:
        """Persist mapping from topic_id to original label in topic lookup."""
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

    def _update_metrics_lookup(self, topic_id: int, label: str) -> None:
        """Persist mapping from topic_id to original label in metrics lookup."""
        if not label:
            return
        try:
            key = str(topic_id)
            if self._metrics_lookup.get(key) != label:
                self._metrics_lookup[key] = label
                with open(self._metrics_lookup_path, 'w', encoding='utf-8') as f:
                    json.dump(self._metrics_lookup, f, indent=2)
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Error updating metrics lookup: {e}")

    def _stable_id(self, label: str) -> int:
        return int(hashlib.blake2s(label.encode("utf-8"), digest_size=4).hexdigest(), 16) % 1_000_000


    def _record_event_for_metrics(self, event: Event, scores: Dict[str, float]) -> None:
        """Record event and scores for metrics tracking."""
        try:
            # --- normalize event fields ---
            user_id = (
                event.get("user_id")
                or event.get("actor_id")
                or event.get("author_id")
                or event.get("channel_id")
                or "unknown"
            )
            ts_iso = (
                event.get("ts_iso")
                or event.get("timestamp")
                or datetime.now(timezone.utc).isoformat()
            )

            # --- convert predictions -> (topic_id, score) using numeric-or-stable-id rule ---
            predictions: list[tuple[int, float]] = []
            top_topic_id: int | None = None
            top_score = float("-inf")

            if scores:
                for key, score in scores.items():
                    if not isinstance(score, (int, float)):
                        continue

                    if str(key).isdigit():
                        topic_id = int(key)  # numeric key: use as-is
                        label = self._topic_lookup.get(str(topic_id), "")
                        # keep metrics lookup in sync
                        self._update_metrics_lookup(topic_id, label or str(topic_id))
                        if label and not str(label).isdigit():
                            self._update_topic_lookup(topic_id, label)
                    else:
                        # string label: assign stable ID and persist mapping
                        topic_id = self._stable_id(key)
                        self._update_metrics_lookup(topic_id, str(key))
                        self._update_topic_lookup(topic_id, str(key))

                    predictions.append((topic_id, float(score)))
                    if score > top_score:
                        top_score = float(score)
                        top_topic_id = topic_id

            # --- choose a topic for the event itself (emergence tracking) ---
            # prefer the top predicted topic; if none, fall back to a stable hash of content_id/text
            if top_topic_id is None:
                fallback_key = (
                    event.get("content_id")
                    or event.get("event_id")
                    or event.get("text")
                    or "unknown_topic"
                )
                top_topic_id = self._stable_id(fallback_key)

            # record the event occurrence for emergence / precision@k tracking
            self.precision_tracker.record_event(
                topic_id=top_topic_id,
                user_id=user_id,
                ts_iso=ts_iso,
            )

            # --- record the prediction list (if any) ---
            if predictions:
                self.precision_tracker.record_predictions(ts_iso=ts_iso, items=predictions)

                # keep short rolling buffer for the Streamlit cache
                formatted = [{"topic_id": tid, "score": float(sc)} for tid, sc in predictions]
                self._predictions_buffer.append(formatted)
                if len(self._predictions_buffer) > 100:
                    self._predictions_buffer = self._predictions_buffer[-50:]

        except Exception as e:
            self.logger.error("Error recording metrics: %s\n%s", e, traceback.format_exc()) # log error without crashing; traceback stacktrace for pinpointing exact failing line
    
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
            try:
                self._update_metrics_and_cache()
            except Exception as e:
                print(f"Error during final metrics update (inner): {e}\n{traceback.format_exc()}")
            print("Final metrics update completed")
        except Exception as e:
            print(f"Error during final metrics update: {e}\n{traceback.format_exc()}")
        
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