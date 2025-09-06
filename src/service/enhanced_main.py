"""Enhanced main service integrating the new data pipeline.

Provides unified entry point that:
- Initializes database and scheduler
- Runs scheduled data collection
- Provides API endpoints for dashboard
- Maintains compatibility with existing components
"""

import os
import sys
import signal
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
except ImportError:
    project_root = Path(__file__).resolve().parents[2]

# Ensure we can import from src
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import new pipeline components
from data_pipeline.storage.database import init_database, get_db_manager
from data_pipeline.scheduler import start_scheduler, stop_scheduler, get_scheduler
from service.api.prediction_api import app as api_app
from utils.logging import setup_logging, get_logger

# Import existing components for compatibility
from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from data_pipeline.storage.builder import GraphBuilder
from model.inference.spam_filter import SpamScorer
from model.inference.adaptive_thresholds import SensitivityController

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global state
_shutdown_event = threading.Event()
_api_server = None
_api_thread = None


class EnhancedPipelineService:
    """Main service class for the enhanced data pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the enhanced pipeline service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.EnhancedPipelineService")
        
        # Component references
        self.db_manager = None
        self.scheduler = None
        self.api_server = None
        
        # Existing components for compatibility
        self.graph_builder = None
        self.embedder = None
        self.spam_scorer = None
        self.sensitivity_controller = None
        
        self.logger.info("Enhanced pipeline service initialized")
    
    def initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")
        
        # Initialize database
        self.logger.info("Setting up database...")
        database_url = self.config.get('database_url')
        self.db_manager = init_database(database_url)
        
        # Initialize existing ML components for compatibility
        self.logger.info("Initializing ML components...")
        self.spam_scorer = SpamScorer()
        self.sensitivity_controller = SensitivityController()
        self.graph_builder = GraphBuilder(spam_scorer=self.spam_scorer)
        self.embedder = RealtimeTextEmbedder(
            batch_size=8, 
            max_latency_ms=50, 
            device="cpu"
        )
        
        # Initialize scheduler with configuration
        self.logger.info("Setting up scheduler...")
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler = start_scheduler(scheduler_config)
        
        self.logger.info("All components initialized successfully")
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API server in a separate thread."""
        global _api_server, _api_thread
        
        self.logger.info(f"Starting API server on {host}:{port}")
        
        def run_server():
            import uvicorn
            uvicorn.run(api_app, host=host, port=port, log_level="info")
        
        _api_thread = threading.Thread(target=run_server, daemon=True)
        _api_thread.start()
        
        # Give server time to start
        time.sleep(2)
        self.logger.info("API server started")
    
    def run_immediate_collection(self):
        """Run immediate data collection for testing."""
        if self.scheduler:
            self.logger.info("Running immediate data collection...")
            self.scheduler.run_immediate_collection()
        else:
            self.logger.error("Scheduler not initialized")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall service status."""
        try:
            # Database status
            db_connected = False
            total_events = 0
            try:
                with self.db_manager.get_session() as session:
                    from data_pipeline.storage.database import Event
                    total_events = session.query(Event).count()
                    db_connected = True
            except Exception:
                pass
            
            # Scheduler status
            scheduler_status = {}
            if self.scheduler:
                scheduler_status = self.scheduler.get_status()
            
            return {
                "service": "Enhanced Data Pipeline",
                "status": "running" if db_connected else "degraded",
                "database": {
                    "connected": db_connected,
                    "total_events": total_events
                },
                "scheduler": scheduler_status,
                "components": {
                    "database_manager": self.db_manager is not None,
                    "scheduler": self.scheduler is not None,
                    "graph_builder": self.graph_builder is not None,
                    "embedder": self.embedder is not None,
                    "spam_scorer": self.spam_scorer is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"status": "error", "error": str(e)}
    
    def shutdown(self):
        """Gracefully shutdown the service."""
        self.logger.info("Shutting down enhanced pipeline service...")
        
        # Stop scheduler
        if self.scheduler:
            try:
                stop_scheduler()
                self.logger.info("Scheduler stopped")
            except Exception as e:
                self.logger.error(f"Error stopping scheduler: {e}")
        
        # API server will stop when main thread exits
        self.logger.info("Enhanced pipeline service shutdown complete")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown_event.set()


def main(config_path: Optional[str] = None, mode: str = "full"):
    """Main entry point for the enhanced pipeline service.
    
    Args:
        config_path: Path to YAML configuration file
        mode: Service mode - 'full', 'api_only', 'scheduler_only', 'collect_once'
    """
    logger.info("Starting Enhanced Trend Prediction Pipeline...")
    logger.info(f"Mode: {mode}")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = {}
    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Initialize service
    service = EnhancedPipelineService(config)
    
    try:
        if mode in ["full", "scheduler_only", "collect_once"]:
            # Initialize all components for scheduler modes
            service.initialize_components()
            
            if mode == "collect_once":
                # Run immediate collection and exit
                logger.info("Running immediate data collection...")
                service.run_immediate_collection()
                logger.info("Collection complete, exiting")
                return
        
        if mode in ["full", "api_only"]:
            # Start API server
            api_port = config.get('api_port', 8000)
            service.start_api_server(port=api_port)
        
        if mode == "scheduler_only":
            logger.info("Running in scheduler-only mode")
        
        # Print status and wait for shutdown
        if mode != "collect_once":
            logger.info("Service started successfully")
            status = service.get_status()
            logger.info(f"Service status: {status}")
            
            # Keep running until shutdown signal
            while not _shutdown_event.is_set():
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        # Cleanup
        service.shutdown()
        logger.info("Service shutdown complete")


def run_api_server():
    """Standalone function to run just the API server."""
    logger.info("Starting API server only...")
    
    # Initialize database
    init_database()
    
    # Start scheduler
    start_scheduler()
    
    # Run API server
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)


def run_scheduler():
    """Standalone function to run just the scheduler."""
    logger.info("Starting scheduler only...")
    
    # Initialize database
    init_database()
    
    # Start scheduler
    scheduler = start_scheduler()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        stop_scheduler()


def run_collection():
    """Standalone function to run immediate data collection."""
    logger.info("Running immediate data collection...")
    
    # Initialize database
    init_database()
    
    # Get scheduler and run collection
    scheduler = get_scheduler()
    scheduler.run_immediate_collection()
    
    logger.info("Collection complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Trend Prediction Pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "api_only", "scheduler_only", "collect_once"],
                       help="Service mode")
    parser.add_argument("--api-port", type=int, default=8000,
                       help="API server port")
    
    args = parser.parse_args()
    
    # Override config with command line args
    config = {}
    if args.api_port != 8000:
        config['api_port'] = args.api_port
    
    main(config_path=args.config, mode=args.mode)