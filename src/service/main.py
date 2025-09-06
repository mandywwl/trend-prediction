"""Unified main entry point for streaming trend prediction service.

Integrates RuntimeGlue with existing pipeline structures for a complete
streaming service with metrics, caching, and graceful shutdown.
"""

import os
import sys
import threading
import signal
from pathlib import Path
from typing import Dict, Any

# Ensure we can import from src
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Pipeline imports
from data_pipeline.collectors.twitter_collector import fake_twitter_stream
from data_pipeline.collectors.youtube_collector import start_youtube_api_collector
from data_pipeline.collectors.google_trends_collector import start_google_trends_collector, fake_google_trends_stream
from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from data_pipeline.storage.builder import GraphBuilder
from model.inference.spam_filter import SpamScorer
from model.inference.adaptive_thresholds import SensitivityController
from service.services.preprocessing.event_handler import EventHandler
from service.runtime_glue import RuntimeGlue, RuntimeConfig
from utils.io import ensure_dir
from utils.logging import get_logger, service_logger, setup_logging
from config.schemas import Event

# Setup logging
setup_logging()
logger = get_logger(__name__)

try:
    from data_pipeline.processors.preprocessing import build_tgn
except Exception:
    build_tgn = None

# Configuration
DATA_DIR = ensure_dir(project_root / "datasets")

# API Keys (TODO: Move to environment variables)
TWITTER_BEARER_TOKEN = "Your_Twitter_Bearer_Token" 
YOUTUBE_API_KEY = "AIzaSyBCiebLZPuGWg0plQJQ0PP6WbZsv0etacs"
KEYWORDS = ["#trending", "fyp", "viral"]

# Global state for graceful shutdown
shutdown_event = threading.Event()
collectors_running = []
runtime_glue_instance = None


class IntegratedEventHandler:
    """Enhanced event handler that integrates with RuntimeGlue metrics."""
    
    def __init__(self, embedder, graph_builder, spam_scorer, sensitivity_controller):
        self.embedder = embedder
        self.graph_builder = graph_builder
        self.spam_scorer = spam_scorer
        self.sensitivity_controller = sensitivity_controller
        self.logger = get_logger(f"{__name__}.IntegratedEventHandler")
        
        # Create the underlying EventHandler
        def _infer_into_graph(event):
            self.graph_builder.process_event(event)
        
        self.event_handler = EventHandler(
            embedder=embedder,
            infer_fn=_infer_into_graph,
            spam_scorer=spam_scorer,
            sensitivity=sensitivity_controller,
        )
        
        # Event counter for checkpoints
        self.event_counter = 0
    
    def on_event(self, event: Event) -> Dict[str, float]:
        """Process event and return prediction scores for RuntimeGlue."""
        try:
            # Process through existing handler
            self.event_handler.handle(event)
            self.event_counter += 1
            
            # Save periodic checkpoints
            if self.event_counter % 100 == 0:
                self._save_checkpoint()
            
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


def setup_preprocessing():
    """Setup preprocessing if needed."""
    tgn_file = DATA_DIR / "tgn_edges_basic.npz"
    
    if not build_tgn:
        logger.info("build_tgn not available; skipping preprocessing")
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


def create_event_stream():
    """Create a unified event stream from all collectors."""
    import queue
    import time
    
    stream_logger = get_logger(f"{__name__}.event_stream")
    event_queue = queue.Queue()
    collectors = []
    
    def twitter_collector():
        """Twitter collector thread."""
        def on_twitter_event(event):
            if not shutdown_event.is_set():
                event_queue.put(event)
        
        try:
            # Use fake stream for demo (replace with real API call if needed)
            fake_twitter_stream(keywords=KEYWORDS, on_event=on_twitter_event, n_events=50, delay=2.0)
        except Exception as e:
            stream_logger.error(f"Twitter collector error: {e}")
    
    def youtube_collector():
        """YouTube collector thread."""
        def on_youtube_event(event):
            if not shutdown_event.is_set():
                event_queue.put(event)
        
        try:
            start_youtube_api_collector(YOUTUBE_API_KEY, on_event=on_youtube_event)
        except Exception as e:
            stream_logger.error(f"YouTube collector error: {e}")
    
    def trends_collector():
        """Google Trends collector thread."""
        def on_trends_event(event):
            if not shutdown_event.is_set():
                event_queue.put(event)
        
        try:
            # Use fake stream for demo
            fake_google_trends_stream(on_event=on_trends_event, n_events=20, delay=5.0)
        except Exception as e:
            stream_logger.error(f"Trends collector error: {e}")
    
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
