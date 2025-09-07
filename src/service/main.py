"""Unified main entry point for streaming trend prediction service."""

import os
import sys
import threading
import signal
import time
import json
from pathlib import Path
from typing import Dict, Any, Callable, Sequence, Iterable, Generator, Tuple

import numpy as np

from utils.path_utils import find_repo_root

# Load environment variables from .env file
project_root = find_repo_root()
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    print("Warning: python-dotenv not installed. Please install it with: pip install python-dotenv")

# Ensure we can import from src
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Pipeline imports
from data_pipeline.collectors.twitter_collector import fake_twitter_stream, start_twitter_stream
from data_pipeline.collectors.youtube_collector import start_youtube_api_collector
from data_pipeline.collectors.google_trends_collector import start_google_trends_collector, fake_google_trends_stream
from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from data_pipeline.storage.builder import GraphBuilder
from model.inference.spam_filter import SpamScorer
from model.inference.adaptive_thresholds import SensitivityController
from service.runtime_glue import RuntimeGlue, RuntimeConfig
from utils.io import ensure_dir
from utils.logging import get_logger, service_logger, setup_logging
from config.schemas import Event, Features
from config.config import EMBED_PREPROC_BUDGET_MS

# Setup logging
setup_logging()
logger = get_logger(__name__)

try:
    from data_pipeline.processors.preprocessing import build_tgn
except Exception:
    build_tgn = None

# Configuration
DATA_DIR = ensure_dir(project_root / "datasets")

# API Keys from environment variables
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Validate required environment variables
if not YOUTUBE_API_KEY:
    logger.warning("YOUTUBE_API_KEY not set - YouTube collector may not work")
if not TWITTER_BEARER_TOKEN:
    logger.warning("TWITTER_BEARER_TOKEN not set - Twitter collector may not work")

# Keywords for Twitter stream
KEYWORDS = ["#trending", "fyp", "viral"]

# Global state for graceful shutdown
shutdown_event = threading.Event()
collectors_running = []
runtime_glue_instance = None

# Event logging setup
EVENT_LOG = DATA_DIR / "events.jsonl"
_event_log_lock = threading.Lock()
# Ensure the parent directory exists before opening the file
EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
_event_log_fh = open(EVENT_LOG, "a", encoding="utf-8")
_MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB


def _rotate_event_log() -> None:
    """Rotate the event log when it grows too large."""
    global _event_log_fh
    rotated = EVENT_LOG.with_name(f"events_{int(time.time())}.jsonl")
    _event_log_fh.close()
    os.rename(EVENT_LOG, rotated)
    _event_log_fh = open(EVENT_LOG, "a", encoding="utf-8")


def log_event(event: Event) -> None:
    """Append ``event`` to the JSONL event log in a thread-safe manner."""
    global _event_log_fh
    with _event_log_lock:
        json.dump(event, _event_log_fh)
        _event_log_fh.write("\n")
        _event_log_fh.flush()
        if _event_log_fh.tell() >= _MAX_LOG_BYTES:
            _rotate_event_log()


# Database storage
DATABASE_PATH = DATA_DIR / "events.db"
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
    # Store in JSONL (for backwards compatibility)
    log_event(event)
    
    # Store in database (for new features)
    store_event_in_database(event)


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

        self._zero_emb: np.ndarray | None = None

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
            processed.setdefault("features", {})["edge_weight"] = weight

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
    def __init__(self, embedder, graph_builder, spam_scorer, sensitivity_controller):
        # Create the inference function for the parent EventHandler
        def _infer_into_graph(event):
            graph_builder.process_event(event)
        
        # Initialize parent EventHandler
        super().__init__(
            embedder=embedder,
            infer=_infer_into_graph,
            spam_scorer=spam_scorer,
            sensitivity=sensitivity_controller,
        )
        
        self.graph_builder = graph_builder
        self.logger = get_logger(f"{__name__}.IntegratedEventHandler")
        
        # Event counter for checkpoints
        self.event_counter = 0
    
    def on_event(self, event: Event) -> Dict[str, float]:
        """Process event and return prediction scores for RuntimeGlue."""
        try:
            # Process through parent handler
            super().handle(event)
            self.event_counter += 1
            
            # Save periodic checkpoints
            if self.event_counter % 100 == 0:
                self._save_checkpoint()
            
            # Run periodic preprocessing (every 1000 events to avoid overhead)
            if self.event_counter % 1000 == 0:
                periodic_preprocessing()
            
            # Generate mock prediction scores for now
            # TODO: Replace with actual TGN inference when available
            scores = {
                f"topic_{i}": 0.8 - (i * 0.1) 
                for i in range(5)
            }
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            return {}
    
    def _save_checkpoint(self):
        """Save graph checkpoint."""
        try:
            checkpoint_path = DATA_DIR / f"checkpoint_{self.event_counter}.pt"
            import torch
            torch.save(self.graph_builder.to_temporal_data(), checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")


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
    events_file = EVENT_LOG
    
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
            build_tgn()
        else:
            logger.info("TGN file exists, skipping preprocessing")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        logger.info("Continuing without preprocessing - collectors will still run")


def periodic_preprocessing():
    """Attempt preprocessing periodically after events have been collected."""
    events_file = EVENT_LOG
    tgn_file = DATA_DIR / "tgn_edges_basic.npz"
    
    if not build_tgn:
        return
    
    # Only attempt if events file exists and has some content
    if events_file.exists() and events_file.stat().st_size > 0:
        try:
            # Check if we should rebuild (weekly or if file doesn't exist)
            should_rebuild = not tgn_file.exists()
            if tgn_file.exists():
                # Rebuild weekly (604800 seconds = 1 week)
                file_age = time.time() - tgn_file.stat().st_mtime
                should_rebuild = file_age > 604800
            
            if should_rebuild:
                logger.info("Running periodic preprocessing (build_tgn)...")
                build_tgn()
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
        """Twitter collector thread.
        Tries real Twitter stream first; falls back to fake stream if credentials
        missing or an error occurs.
        """
        def on_twitter_event(event):
            if not shutdown_event.is_set():
                enhanced_log_event(event)
                event_queue.put(event)

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
                stream_logger.error(f"Real Twitter stream failed, falling back to fake: {e}")

        # Fallback: simulated events (finite)
        try:
            fake_twitter_stream(
                keywords=KEYWORDS,
                on_event=on_twitter_event,
                n_events=50,
                delay=2.0,
            )
        except Exception as e:
            stream_logger.error(f"Twitter collector (fake fallback) error: {e}")
    
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
        """Google Trends collector thread.
        Attempts real pytrends polling; falls back to synthetic if pytrends
        unavailable or an error occurs.
        """
        def on_trends_event(event):
            if not shutdown_event.is_set():
                enhanced_log_event(event)
                event_queue.put(event)

        try:
            # Real collector (blocks with internal loop)
            start_google_trends_collector(
                on_event=on_trends_event,
                region="US",
                category="all",
                count=20,
                interval=3600,
            )
            return
        except Exception as e:
            stream_logger.error(f"Real Google Trends collector failed, falling back to fake: {e}")

        # Fallback: finite synthetic events
        try:
            fake_google_trends_stream(on_event=on_trends_event, n_events=20, delay=5.0)
        except Exception as e:
            stream_logger.error(f"Trends collector (fake fallback) error: {e}")
    
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


def main(yaml_config_path: str = None):
    """Main entry point with RuntimeGlue integration."""
    global runtime_glue_instance
    
    logger.info("Starting unified trend prediction service...")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup preprocessing
    setup_preprocessing()
    
    # Initialize training scheduler and hourly collector
    logger.info("Initializing training scheduler and hourly collector...")
    training_scheduler = None
    hourly_collector = None
    
    if event_database:
        try:
            from service.training_scheduler import TrainingScheduler, HourlyDataCollector
            
            # Initialize training scheduler (weekly retraining)
            training_scheduler = TrainingScheduler(
                database=event_database,
                datasets_dir=DATA_DIR,
                training_interval_hours=168,  # 1 week
                min_events_for_training=100
            )
            
            # Initialize hourly data collector
            hourly_collector = HourlyDataCollector(database=event_database)
            
            # Start background services
            training_scheduler.start()
            hourly_collector.start()
            
            logger.info("Training scheduler and hourly collector started")
        except Exception as e:
            logger.error(f"Failed to initialize training components: {e}")
    
    # Initialize components
    logger.info("Initializing components...")
    spam_scorer = SpamScorer()
    graph_builder = GraphBuilder(spam_scorer=spam_scorer)
    sensitivity_controller = SensitivityController()
    embedder = RealtimeTextEmbedder(batch_size=8, max_latency_ms=50, device="cpu")
    
    # Create integrated event handler
    event_handler = IntegratedEventHandler(
        embedder=embedder,
        graph_builder=graph_builder,
        spam_scorer=spam_scorer,
        sensitivity_controller=sensitivity_controller
    )
    
    # Configure RuntimeGlue
    config = RuntimeConfig.from_yaml(yaml_config_path)
    runtime_glue = RuntimeGlue(event_handler, config)
    runtime_glue_instance = runtime_glue  # Store for signal handler
    
    logger.info(f"Configuration: {config.__dict__}")
    
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
