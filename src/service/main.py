"""Unified main entry point for streaming trend prediction service."""

import os
import sys
import threading
import signal
import time
import json
import numpy as np
import random

from pathlib import Path
from typing import Dict, Any, Callable, Sequence, Iterable, Generator, Tuple
from datetime import datetime, timezone
from service.tgn_service import TGNInferenceService
from service.runtime_glue import RuntimeGlue, RuntimeConfig
from model.inference.spam_filter import SpamScorer
from model.inference.adaptive_thresholds import SensitivityController

from utils.logging import get_logger, setup_logging
from config.config import (
    PROJECT_ROOT,
    TOPIC_LOOKUP_PATH,
    EVENT_JSONL_PATH,
    INFERENCE_DEVICE,
    DATA_DIR,
    DATABASE_PATH,
    KEYWORDS,
    DEFAULT_TRENDING_TOPICS,
    TWITTER_SIM_EVENTS_PER_BATCH,
    TWITTER_SIM_BATCH_INTERVAL,
    REGIONS,
    TRENDS_CATEGORY,
    TRENDS_COUNT,
    TRENDS_INTERVAL_SEC,
    EVENT_MAX_LOG_BYTES,
    EMBEDDER_BATCH_SIZE,
    EMBEDDER_MAX_LATENCY_MS,
    EMBED_PREPROC_BUDGET_MS,
    TRAINING_INTERVAL_HOURS,
    MIN_EVENTS_FOR_TRAINING,
    TOPIC_REFRESH_EVERY,
    TOPIC_REFRESH_SECS,
    PERIODIC_REBUILD_SECS,
    EDGE_WEIGHT_MIN,

)
from config.schemas import Event, Features
from data_pipeline.collectors.twitter_collector import start_twitter_stream, realistic_fake_twitter_stream, enhanced_fake_twitter_stream
from data_pipeline.collectors.youtube_collector import start_youtube_api_collector
from data_pipeline.collectors.google_trends_collector import start_google_trends_collector, fake_google_trends_stream
from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from data_pipeline.storage.builder import GraphBuilder


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("Warning: python-dotenv not installed. Please install it with: pip install python-dotenv")

# Ensure we can import from src
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Setup logging
setup_logging()
logger = get_logger(__name__)
try:
    from data_pipeline.processors.preprocessing import build_tgn
except Exception:
    build_tgn = None

# Get API Keys from environment variables
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Validate required environment variables
if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY not set - YouTube collector may not work")
if not TWITTER_BEARER_TOKEN:
    logger.warning("TWITTER_BEARER_TOKEN not set - Twitter collector may not work")
if not KEYWORDS:
    logger.warning("KEYWORDS not set - Twitter/TikTok collector may not work")


shutdown_event = threading.Event() # Global state for graceful shutdown
collectors_running = []
runtime_glue_instance = None
_event_log_lock = threading.Lock()
EVENT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure the parent directory exists before opening the file
_event_log_fh = open(EVENT_JSONL_PATH, "a", encoding="utf-8")


def _rotate_event_log() -> None:
    """Rotate the event log when it grows too large."""
    global _event_log_fh
    rotated = EVENT_JSONL_PATH.with_name(f"events_{int(time.time())}.jsonl")
    _event_log_fh.close()
    os.rename(EVENT_JSONL_PATH, rotated)
    _event_log_fh = open(EVENT_JSONL_PATH, "a", encoding="utf-8")


def log_event(event: Event) -> None:
    """Append ``event`` to the JSONL event log in a thread-safe manner."""
    global _event_log_fh
    _MAX_LOG_BYTES = int(EVENT_MAX_LOG_BYTES)
    with _event_log_lock:
        json.dump(event, _event_log_fh)
        _event_log_fh.write("\n")
        _event_log_fh.flush()
        if _event_log_fh.tell() >= _MAX_LOG_BYTES:
            _rotate_event_log()

# Database storage
try:
    from data_pipeline.storage.database import EventDatabase
    event_database = EventDatabase(DATABASE_PATH)
    logger.info(f"Event database initialized at {DATABASE_PATH}")
except Exception as e:
    logger.warning(f"Failed to initialize database: {e}")
    event_database = None


def store_event_in_database(event: Event) -> None:
    """Store event in database if available."""
    if event_database:
        try:
            event_database.store_event(event)
        except Exception as e:
            logger.error(f"Failed to store event in database: {e}")

def enhanced_log_event(event: Event) -> None:
    """Enhanced event logging that stores in both JSONL and database."""
    log_event(event) # Store in JSONL for checking
    store_event_in_database(event) # Store in database


class EmbeddingPreprocessor:
    """Attach text embeddings to incoming events.

    The preprocessor looks for known text fields, encodes the first one found
    using :class:`RealtimeTextEmbedder` and stores the resulting vector under
    ``event["features"]["text_emb"]``.
    """

    def __init__(
        self,
        embedder: RealtimeTextEmbedder,
        *,
        text_fields: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            embedder: Instance of :class:`RealtimeTextEmbedder` used for
                generating embeddings.
            text_fields: Optional list of event keys to inspect for text. The
                first present key is used. Defaults to common fields such as
                ``"text"``, ``"tweet_text"`` and ``"caption"``.
        """
        self.embedder = embedder
        self.text_fields = (
            list(text_fields)
            if text_fields is not None
            else [
                "text",
                "tweet_text",
                "caption",
                "description",
            ]
        )

        self._zero_emb = None  # Will be np.ndarray when initialized

    def _extract_text(self, event: Event) -> str | None:
        for field in self.text_fields:
            value = event.get(field)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def __call__(self, event: Event, *, light: bool = False) -> Event:
        """Process ``event`` in-place and return it.

        If no text field is found a zero embedding is attached to satisfy the
        downstream schema.
        """
        t0 = time.perf_counter()
        text = self._extract_text(event)
        if text is None or light:
            if self._zero_emb is None:
                import numpy as np
                dim = self.embedder.model.config.hidden_size
                self._zero_emb = np.zeros(dim, dtype=np.float32)
            emb = self._zero_emb
        else:
            emb = self.embedder.encode([text])[0]

        features: Features = event.setdefault("features", {})
        features["text_emb"] = emb

        duration_ms = (time.perf_counter() - t0) * 1000.0
        if duration_ms > EMBED_PREPROC_BUDGET_MS:
            print(
                f"[EmbeddingPreprocessor] budget exceeded: {duration_ms:.1f}ms > {EMBED_PREPROC_BUDGET_MS}ms"
            )
        return event


class EventHandler:
    """Handle events by preprocessing then invoking TGN inference."""

    def __init__(
        self,
        embedder: RealtimeTextEmbedder,
        infer: Callable[[Dict[str, Any]], Any],
        *,
        spam_scorer: SpamScorer | None = None,
        sensitivity: "SensitivityController | None" = None,
        service: Any | None = None,
    ) -> None:
        """Initialise the handler.

        Args:
            embedder: Text embedder used by the ``EmbeddingPreprocessor``.
            infer: Callable representing the TGN inference service. It receives
                the processed event.
            spam_scorer: Optional :class:`SpamScorer` used to down-weight
                edges for suspected spammy accounts.
        """
        self.preprocessor = EmbeddingPreprocessor(embedder)
        self._infer = infer
        self.spam_scorer = spam_scorer
        self.sensitivity = sensitivity
        self._service = service  # Optional TGN-like scoring service

    def handle(self, event: Event) -> Any:
        """Preprocess ``event`` and forward it to inference.

        If a sensitivity controller is provided and currently applying
        back-pressure, heavy operations such as embeddings are skipped by
        attaching a zero-vector. The handler also records observed latency
        into the controller for adaptive behaviour.
        """

        policy = None
        if self.sensitivity is not None:
            policy = self.sensitivity.policy()
        light = bool(policy and not policy.heavy_ops_enabled)

        t0 = time.perf_counter()
        processed = self.preprocessor(event, light=light)

        if self.spam_scorer is not None:
            weight = self.spam_scorer.edge_weight(processed)
            processed["edge_weight"] = weight
            processed['is_spam'] = bool(weight < 0.8)  # expose bool spam flag for metrics/robustness panels
            processed.setdefault("features", {})["edge_weight"] = weight

        # recency × spam (half-life 1 hour)
        try:
            ts_iso = processed.get("ts_iso") or processed.get("timestamp")
            # robust parse (no external dep): allow 'Z'
            if isinstance(ts_iso, str):
                ts_iso = ts_iso.replace('Z', '+00:00')
                dt = datetime.fromisoformat(ts_iso)
            else:
                dt = datetime.now(timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
            recency = 0.5 ** delta_hours
        except Exception:
            recency = 1.0

        w_spam = float(processed["features"].get("edge_weight", 1.0))
        combined = max(float(EDGE_WEIGHT_MIN), float(recency) * w_spam)
        processed["features"]["edge_weight"] = combined # what service appends

        result = self._infer(processed)

        # Record latency and (optional) ground-truth spam label for adaptation
        if self.sensitivity is not None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            is_spam = bool(processed.get("is_spam", False))
            try:
                self.sensitivity.record_event(is_spam=is_spam, latency_ms=latency_ms)
            except Exception:
                pass  # never let adaptation interfere with the hot path

            # Expose suggested sampler size for downstream components
            pol = self.sensitivity.policy()
            processed.setdefault("features", {})["sampler_size"] = pol.sampler_size

        return result

    def on_event(self, event: Event) -> Dict[str, float]:
        """Thin path: call scoring service once per event.

        Keeps the handler stateless across runs; temporal state (if any)
        belongs to the injected service.
        """
        if self._service is None:
            # Fallback: run existing pipeline without scores
            self.handle(event)
            return {}
        return self._service.update_and_score(event)


class IntegratedEventHandler(EventHandler):
    """Event handler integrating preprocessing, spam scoring, and TGN graph building.
    
    Extends EventHandler with graph building and checkpoint functionality.
    """
    def __init__(self, embedder, graph_builder, spam_scorer, sensitivity_controller, service=None):
        # Create the inference function for the parent EventHandler
        def _infer_into_graph(event):
            graph_builder.process_event(event)
        
        # Initialize parent EventHandler
        super().__init__(
            embedder=embedder,
            infer=_infer_into_graph,
            spam_scorer=spam_scorer,
            sensitivity=sensitivity_controller,
            service=service, # keep a reference for on_event
        )
        self.runtime_glue = None
        
        self.graph_builder = graph_builder
        self.logger = get_logger(f"{__name__}.IntegratedEventHandler")
        
        # Event counter for checkpoints & last refresh time
        self.event_counter = 0
        self._last_topic_refresh = None
        
        # Load existing topic lookup for generating realistic scores
        self._load_topic_lookup()
        
        # Latency measurement
        from utils.io import LatencyAggregator
        self.latency_aggregator = LatencyAggregator()
    
    def on_event(self, event: Event) -> Dict[str, float]:
        """Process event and return prediction scores for RuntimeGlue."""
        from utils.io import LatencyTimer
        with LatencyTimer() as timer:
            timer.start_stage('ingest')

            # Event validation/logging
            self.event_counter += 1
            timer.end_stage('ingest')

            timer.start_stage('preprocess')
            super().handle(event) # embeddings + spam weight
            timer.end_stage('preprocess')

            timer.start_stage('model_update_forward')
            if self._service is not None:
                # --- Real model inference ---
                tgn_out = self._service.update_and_score(event)  # returns {'growth_score', 'score', ...}
                # Persist raw TGN metrics on the event for downstream consumers (e.g., dashboard panes)
                event.setdefault("tgn_metrics", {}).update(tgn_out)

                # For RuntimeGlue’s P@K, return a topic->score map.
                from service.runtime_glue import RuntimeGlue  # for stable id hashing via glue if needed
                topic_label = event.get("text") or event.get("content_id") or "unknown_topic"
                growth = tgn_out.get("growth_score", tgn_out.get("score", tgn_out.get("growth_rate", 0.0)))
                scores = {str(topic_label): float(growth)}
            else:
                # Fallback: keep the current simulated scores while model is not ready
                scores = self._generate_realistic_scores(event)

            timer.end_stage('model_update_forward')
            timer.start_stage('postprocess')

            # Run periodic preprocessing
            if self.event_counter % 1000 == 0:
                periodic_preprocessing()

            REFRESH_EVERY = TOPIC_REFRESH_EVERY
            REFRESH_SECS  = TOPIC_REFRESH_SECS
            now = datetime.now(timezone.utc)
            backstop_due = (
                not hasattr(self, "_last_topic_refresh")
                or (self._last_topic_refresh is None)
                or ((now - getattr(self, "_last_topic_refresh")).total_seconds() >= REFRESH_SECS)
            )
            if (self.event_counter % REFRESH_EVERY == 0) or backstop_due:
                rg = getattr(self, "runtime_glue", None)
                if rg is not None and not getattr(rg, "is_shutting_down", False):
                    try:
                        rg.update_topic_labels()
                        self._last_topic_refresh = now
                    except Exception as e:
                        self.logger.warning(f"Skipped topic label refresh (backstop): {e}")
            
            # close the preprocess stage
            timer.end_stage('postprocess')

        self._record_latency(timer)
        return scores
    
    def _record_latency(self, timer):
        """Record latency measurements."""
        try:
            self.latency_aggregator.add_measurement(
                timer.total_duration_ms,
                timer.get_stage_ms()
            )
        except Exception as e:
            self.logger.error(f"Error recording latency: {e}")
    
    def _load_topic_lookup(self):
        """Load existing topic lookup for generating realistic scores."""
        self.topic_lookup_path = Path(TOPIC_LOOKUP_PATH)
        self.topic_ids = []
        
        if self.topic_lookup_path.exists():
            try:
                with open(self.topic_lookup_path, 'r', encoding='utf-8') as f:
                    topic_mapping = json.load(f)
                    # Get all existing topic IDs whose labels are non-numeric
                    self.topic_ids = [
                        int(tid) for tid, 
                        label in topic_mapping.items() if not str(label).isdigit()
                    ]
                    self.topic_labels = [
                        label for _, 
                        label in topic_mapping.items() if not str(label).isdigit()
                    ]
                    self.logger.info(
                        f"Loaded {len(self.topic_ids)} existing topic IDs from topic_lookup.json"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load topic lookup: {e}")
        
        # Fallback: use some default topic IDs if lookup file is empty/missing
        if not self.topic_ids:
            self.topic_ids = [100000 + i for i in range(10)]  # Generate some default IDs
            self.topic_labels = [f"General {i}" for i in range(10)]  # fallback labels
    
    def _generate_realistic_scores(self, event: Event) -> Dict[str, float]:
        
        labels = getattr(self, "topic_labels", []) or [str(tid) for tid in self.topic_ids]
        k = min(5, len(labels)) or 5
        chos = random.sample(labels, k) if labels else [f"Topic {i}" for i in range(k)]
        base = np.sort(np.random.exponential(scale=0.3, size=k))[::-1]
        scores = {chos[i]: float(min(0.9, max(0.1, base[i]))) for i in range(k)}
        
        return scores
    
    def _refresh_topic_labels(self):
        """Refresh topic labels using the topic labeling pipeline."""
        try:
            from data_pipeline.processors.topic_labeling import run_topic_labeling_pipeline
            
            self.logger.info("Refreshing topic labels...")
            
            # Run the pipeline to generate meaningful labels
            result = run_topic_labeling_pipeline(
                events_path="data/events.jsonl",
                topic_lookup_path=str(self.topic_lookup_path),
                use_embedder=False  # Use TF-IDF only for better performance
            )
            
            # Reload the topic IDs after update
            self._load_topic_lookup()
            
            updated_count = sum(1 for label in result.values() 
                              if not (label.startswith("topic_") or 
                                     label.startswith("test_") or 
                                     label.startswith("viral_") or 
                                     label.startswith("trending_")))
            
            self.logger.info(f"Topic labeling refreshed. Updated {updated_count} labels.")
            
        except Exception as e:
            self.logger.error(f"Error refreshing topic labels: {e}")
    
    def _save_checkpoint(self):
        """Save graph checkpoint."""
        try:
            checkpoint_path = DATA_DIR / f"checkpoint_{self.event_counter}.pt"
            import torch
            torch.save(self.graph_builder.to_temporal_data(), checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

# ----------------------------------------------------------------------
def run_stream(
    reader: Iterable[Event],
    handler: EventHandler,
) -> Generator[Tuple[Event, Dict[str, float]], None, None]:
    """Yield (event, scores) pairs from a synchronous reader.

    Calls the handler's ``on_event`` exactly once per input event.
    """
    for event in reader:
        yield event, handler.on_event(event)


def setup_preprocessing():
    """Setup preprocessing if needed."""
    tgn_file = DATA_DIR / "tgn_edges_basic.npz"
    events_file = EVENT_JSONL_PATH
    
    if not build_tgn:
        logger.info("build_tgn not available; skipping preprocessing")
        return
    
    # Check if events.jsonl exists before attempting preprocessing
    if not events_file.exists():
        logger.info("events.jsonl does not exist yet; skipping preprocessing for now")
        logger.info("Preprocessing will be available after collectors generate some data")
        return
    
    try:
        force_rebuild = os.environ.get("PREPROCESS_FORCE") == "1"
        if not tgn_file.exists() or force_rebuild:
            logger.info("Running preprocessing (build_tgn)...")
            build_tgn(
                events_path=str(events_file),
                output_path=str(tgn_file),
                force=True
            )

        else:
            logger.info("TGN file exists, skipping preprocessing")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        logger.info("Continuing without preprocessing - collectors will still run")


def periodic_preprocessing():
    """Attempt preprocessing periodically after events have been collected."""
    events_file = EVENT_JSONL_PATH
    tgn_file = DATA_DIR / "tgn_edges_basic.npz"
    
    if not build_tgn:
        return
    
    # Only attempt if events file exists and has some content
    if events_file.exists() and events_file.stat().st_size > 0:
        try:
            # Check if we should rebuild (weekly or if file doesn't exist)
            should_rebuild = not tgn_file.exists()
            if tgn_file.exists():
                # Rebuild if older than PERIODIC_REBUILD_SECS (default 1 week)
                file_age = time.time() - tgn_file.stat().st_mtime
                should_rebuild = file_age > PERIODIC_REBUILD_SECS
            
            if should_rebuild:
                logger.info("Running periodic preprocessing (build_tgn)...")
                build_tgn(
                    events_path=str(events_file),
                    output_path=str(tgn_file),
                    force=True
                )
                logger.info("Periodic preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Periodic preprocessing failed: {e}")
    else:
        logger.debug("No events data available yet for preprocessing")


def create_event_stream():
    """Create a unified event stream from all collectors."""
    import queue
    import time
    
    stream_logger = get_logger(f"{__name__}.event_stream")
    event_queue = queue.Queue()
    collectors = []
    

    def twitter_collector():
        """Twitter collector thread with enhanced realistic simulation.
        Tries real Twitter stream first; falls back to enhanced realistic simulation.
        """
        def on_twitter_event(event):
            if not shutdown_event.is_set():
                enhanced_log_event(event)
                event_queue.put(event)

        # Get trending topics from other collectors to influence tweets
        def get_current_trending_topics():
            """Extract recent trending topics to influence tweet content."""
            trending_topics = []
            
            # Try to get recent trends from events.jsonl
            try:
                if EVENT_JSONL_PATH.exists():
                    with open(EVENT_JSONL_PATH, 'r') as f:
                        recent_lines = f.readlines()[-20:]  # Last 20 events
                        
                    for line in recent_lines:
                        try:
                            event = json.loads(line)
                            if event.get('source') == 'google_trends':
                                topic = event.get('text', '')
                                if topic and len(topic) < 50:  # Reasonable length
                                    trending_topics.append(topic)
                        except:
                            continue
            except:
                pass
            
            # Add some default trending topics if none found
            if not trending_topics:
                trending_topics = DEFAULT_TRENDING_TOPICS
            
            return trending_topics[:10]  # Return top 10

        # Prefer real stream when we have a bearer token
        if TWITTER_BEARER_TOKEN:
            try:
                start_twitter_stream(
                    bearer_token=TWITTER_BEARER_TOKEN,
                    keywords=KEYWORDS,
                    on_event=on_twitter_event,
                )
                return
            except Exception as e:
                stream_logger.error(f"Real Twitter stream failed, falling back to enhanced simulation: {e}")

        # Enhanced realistic simulation fallback
        try:
            stream_logger.info("Starting enhanced realistic Twitter simulation...")
            
            # Get trending topics to influence tweet content
            trending_topics = get_current_trending_topics()
            
            # Import the enhanced function
            from data_pipeline.collectors.twitter_collector import enhanced_fake_twitter_stream
            
            enhanced_fake_twitter_stream(
                keywords=KEYWORDS,
                on_event=on_twitter_event,
                events_per_batch=TWITTER_SIM_EVENTS_PER_BATCH,
                batch_interval=TWITTER_SIM_BATCH_INTERVAL,
                topic_hints=trending_topics
            )
            
        except Exception as e:
            stream_logger.error(f"Enhanced Twitter simulation failed: {e}")
            
    
    def youtube_collector():
        """YouTube collector thread."""
        def on_youtube_event(event):
            if not shutdown_event.is_set():
                enhanced_log_event(event)
                event_queue.put(event)
        
        try:
            start_youtube_api_collector(YOUTUBE_API_KEY, on_event=on_youtube_event)
        except Exception as e:
            stream_logger.error(f"YouTube collector error: {e}")
    
    def trends_collector():

        """ Simulated Google Trends collector. """ #TODO: Replace with real API integration when available.

        def on_trends_event(event):
            if not shutdown_event.is_set():
                enhanced_log_event(event)
                event_queue.put(event)

        print("[Trends] Starting enhanced simulated trends collector...")
        
        try:
            # Use realistic simulated trends with 30-minute cycles
            start_google_trends_collector(
                on_event=on_trends_event,
                region=REGIONS[0],  # Uses the first region from config #TODO: Multi-region support
                category=TRENDS_CATEGORY,
                count=TRENDS_COUNT,
                interval=TRENDS_INTERVAL_SEC,
            )
        except Exception as e:
            stream_logger.error(f"Simulated trends collector error: {e}")
            # Fallback to simple fake trends
            while not shutdown_event.is_set():
                try:
                    fake_google_trends_stream(
                        on_event=on_trends_event,
                        n_events=5,
                        delay=2.0
                    )
                    time.sleep(600)  # 10 minutes between batches
                except Exception as fallback_error:
                    stream_logger.error(f"Fallback trends error: {fallback_error}")
                    time.sleep(120)
    
    # Start collector threads
    twitter_thread = threading.Thread(target=twitter_collector, name="TwitterCollector")
    youtube_thread = threading.Thread(target=youtube_collector, name="YouTubeCollector")
    trends_thread = threading.Thread(target=trends_collector, name="TrendsCollector")
    
    collectors.extend([twitter_thread, youtube_thread, trends_thread])
    collectors_running.extend(collectors)
    
    for thread in collectors:
        thread.daemon = True
        thread.start()
        stream_logger.info(f"Started {thread.name}")
    
    # Event stream generator
    def event_generator():
        while not shutdown_event.is_set():
            try:
                # Get event with timeout to allow checking shutdown
                event = event_queue.get(timeout=1.0)
                yield event
                event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                stream_logger.error(f"Event stream error: {e}")
                break
    
    return event_generator()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()
    # Also signal the runtime glue if it exists
    if runtime_glue_instance is not None:
        runtime_glue_instance.set_shutdown()

# ----------------------------------------------------------------------

def main(yaml_config_path: str = None):
    """Main entry point with RuntimeGlue integration."""

    setup_preprocessing()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize components
    logger.info("Initializing components...")
    spam_scorer = SpamScorer()
    graph_builder = GraphBuilder(spam_scorer=spam_scorer)
    sensitivity_controller = SensitivityController()

    embedder = RealtimeTextEmbedder(
        batch_size=EMBEDDER_BATCH_SIZE, 
        max_latency_ms=EMBEDDER_MAX_LATENCY_MS, 
        device=INFERENCE_DEVICE if INFERENCE_DEVICE else "cpu"
    )

    # Try to load a trained checkpoint if present
    ckpt_path = DATA_DIR / "tgn_model.pt"
    tgn_service = None
    try:
        if ckpt_path.exists():
            tgn_service = TGNInferenceService(
                checkpoint_path=str(ckpt_path),
                device=INFERENCE_DEVICE if INFERENCE_DEVICE else "cpu",
                log_dir=str(DATA_DIR)
            )
            logger.info(f"Loaded TGN checkpoint: {ckpt_path}")
        else:
            logger.warning(f"No TGN checkpoint at {ckpt_path}; running with service=None until trained.")
    except Exception as e:
        logger.error(f"Failed to initialize TGNInferenceService: {e}")

    # Create integrated event handler with live TGN service
    event_handler = IntegratedEventHandler(
        embedder=embedder,
        graph_builder=graph_builder,
        spam_scorer=spam_scorer,
        sensitivity_controller=sensitivity_controller,
        service=tgn_service,
    )


    def _reload_ckpt(p: Path) -> None:
        """Hot-reload (or late-init) the TGN service after retraining."""
        try:
            nonlocal tgn_service, event_handler
            if tgn_service is None:
                # late init on first checkpoint
                tgn_service = TGNInferenceService(checkpoint_path=str(p), device="cpu", log_dir=str(DATA_DIR))
                event_handler._service = tgn_service          # ensure handler uses it
                event_handler.tgn_service = tgn_service        # if you keep this alias elsewhere
                logger.info(f"Initialized TGN service with checkpoint: {p}")
            else:
                # hot reload
                if hasattr(tgn_service, "reload_checkpoint"):
                    tgn_service.reload_checkpoint(str(p))
                else:
                    # fallback if method not implemented
                    tgn_service = TGNInferenceService(checkpoint_path=str(p), device="cpu", log_dir=str(DATA_DIR))
                    event_handler._service = tgn_service
                    event_handler.tgn_service = tgn_service
                logger.info(f"Reloaded TGN checkpoint: {p}")
        except Exception as e:
            logger.error(f"Failed to (re)load TGN checkpoint: {e}")

    
    # Run initial topic labeling to ensure meaningful labels exist
    logger.info("Running initial topic labeling pipeline...")
    try:
        event_handler._refresh_topic_labels()
    except Exception as e:
        logger.warning(f"Initial topic labeling failed: {e}")
    
    # Configure RuntimeGlue
    runtime_config = RuntimeConfig.from_yaml(yaml_config_path)
    runtime_glue = RuntimeGlue(event_handler, runtime_config)
    event_handler.runtime_glue = runtime_glue  # allow handler to call update_topic_labels() safely later
    global runtime_glue_instance
    runtime_glue_instance = runtime_glue  # Store for signal handler

    logger.info(f"Configuration: {runtime_config.__dict__}")

    training_scheduler = None
    hourly_collector = None
    if event_database:
        try:
            from service.training_scheduler import TrainingScheduler, HourlyDataCollector
            
            # Initialize training scheduler
            training_scheduler = TrainingScheduler(
                database=event_database,
                datasets_dir=DATA_DIR,
                training_interval_hours=TRAINING_INTERVAL_HOURS,
                min_events_for_training=MIN_EVENTS_FOR_TRAINING,
                on_new_checkpoint=_reload_ckpt,
            )
            
            # Initialize hourly data collector
            hourly_collector = HourlyDataCollector(database=event_database)
            
            # Start background services
            training_scheduler.start()
            hourly_collector.start()
            
            logger.info("Training scheduler and hourly collector started")
        except Exception as e:
            logger.error(f"Failed to initialize training components: {e}")
    
    # Variables for cleanup
    training_scheduler_ref = training_scheduler
    hourly_collector_ref = hourly_collector
    try:
        # Create unified event stream
        logger.info("Starting data collectors...")
        event_stream = create_event_stream()
        
        # Run the streaming service
        logger.info("Starting streaming service with RuntimeGlue...")
        runtime_glue.run_stream(event_stream)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        # Cleanup
        logger.info("Performing cleanup...")
        shutdown_event.set()
        
        # Stop training scheduler and hourly collector
        if training_scheduler_ref:
            training_scheduler_ref.stop()
        if hourly_collector_ref:
            hourly_collector_ref.stop()
        
        # Wait for collector threads to finish
        for thread in collectors_running:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Final checkpoint
        try:
            final_checkpoint = DATA_DIR / "final_checkpoint.pt"
            import torch
            torch.save(event_handler.graph_builder.to_temporal_data(), final_checkpoint)
            logger.info(f"Saved final checkpoint: {final_checkpoint}")
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}")

        # Close event log file
        try:
            _event_log_fh.close()
        except Exception:
            pass

        logger.info("Service shutdown complete.")


if __name__ == "__main__":
    import sys
    
    # Activate virtual environment message
    print("Please ensure your virtual environment is activated:")
    print("Windows: .venv\\Scripts\\activate")
    print("Linux/Mac: source .venv/bin/activate")
    print()
    
    # Get optional YAML config path
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else None
    if yaml_path:
        print(f"Using config file: {yaml_path}")
    
    main(yaml_path)
