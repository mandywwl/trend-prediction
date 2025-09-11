"""Training scheduler for periodic model retraining."""

import time
import threading
import json
import tempfile
import os

from datetime import datetime, timezone, timedelta
from pathlib import Path
import traceback
from typing import Optional, Callable, Any

from utils.logging import get_logger
from utils.datetime import utc_now
from data_pipeline.storage.database import EventDatabase
from model.training.train import train_tgn_from_npz
from config.config import (
    DATA_DIR,
    TRAINING_INTERVAL_HOURS,
    MIN_EVENTS_FOR_TRAINING,
    TRAIN_EPOCHS,
    SCHEDULER_POLL_SECONDS,
    INFERENCE_DEVICE,
)

logger = get_logger(__name__)


class TrainingScheduler:
    """Handles periodic model retraining based on collected data."""
    
    def __init__(
        self,
        database: EventDatabase,
        datasets_dir: Path,
        training_interval_hours = TRAINING_INTERVAL_HOURS,
        min_events_for_training = MIN_EVENTS_FOR_TRAINING,
        on_new_checkpoint: Optional[Callable[[Path], None]] = None,
    ):
        """Initialize the training scheduler.
        
        Args:
            database: Event database instance
            datasets_dir: Directory for storing training data
            training_interval_hours: Hours between training runs (default: 1 week)
            min_events_for_training: Minimum events needed to trigger training
        """
        self.database = database
        self.datasets_dir = Path(datasets_dir)
        self.training_interval = timedelta(hours=training_interval_hours)
        self.min_events = min_events_for_training
        self.on_new_checkpoint = on_new_checkpoint
        self.last_training_time: Optional[datetime] = None
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info(f"Training scheduler initialized (interval: {training_interval_hours}h, min_events: {min_events_for_training})")
    
    def start(self) -> None:
        """Start the training scheduler in a background thread."""
        if self.is_running:
            logger.warning("Training scheduler is already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logger.info("Training scheduler started")
    
    def stop(self) -> None:
        """Stop the training scheduler."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Training scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop that checks for training conditions."""
        while self.is_running:
            try:
                if self._should_train():
                    self._trigger_training()
                
                # Check every hour
                time.sleep(SCHEDULER_POLL_SECONDS)
                
            except Exception as e:
                logger.error(f"Error in training scheduler: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _should_train(self) -> bool:
        """Check if training should be triggered."""
        now = utc_now()
        
        # Check if enough time has passed since last training
        if self.last_training_time is not None:
            time_since_last = now - self.last_training_time
            if time_since_last < self.training_interval:
                return False
        
        # Check if we have enough events
        event_count = self.database.get_event_count()
        if event_count < self.min_events:
            logger.debug(f"Not enough events for training: {event_count} < {self.min_events}")
            return False
        
        logger.info(f"Training conditions met: {event_count} events, time elapsed since last training")
        return True
    
    def _trigger_training(self) -> None:
        """Trigger model training with current data."""
        try:
            logger.info("Starting model training...")
            
            # Export current data to JSONL for training
            events_file = self.datasets_dir / "events.jsonl"
            event_count = self.database.export_to_jsonl(events_file)
            
            if event_count == 0:
                logger.warning("No events to export for training")
                return
            
            # Run preprocessing to generate TGN data
            self._run_preprocessing(events_file)

            # Train the model
            npz_path = self.datasets_dir / "tgn_edges_basic.npz"
            ckpt_out = self.datasets_dir / "tgn_model.pt"
            if npz_path.exists():
                train_tgn_from_npz(str(npz_path), # uses trainer from model.training.train
                                   str(ckpt_out), 
                                   epochs=TRAIN_EPOCHS, 
                                   device=INFERENCE_DEVICE,
                                   noise_p=None,
                                   noise_seed=0)
                if Path(ckpt_out).exists():
                    logger.info(f"Training completed successfully; saved {ckpt_out}")
                    self.last_training_time = utc_now()
                    # tell the running app to reload weights (main.py provides this)
                    if self.on_new_checkpoint is not None:
                        self.on_new_checkpoint(ckpt_out)
                else:
                    logger.error(f"Training returned but checkpoint missing: {ckpt_out}")
            else:
                logger.error(f"No NPZ at {npz_path}; skipping model training")
            
        except Exception as e:
            logger.error("Training failed: %s\n%s", e, traceback.format_exc())
    
    def _run_preprocessing(self, events_file: Path) -> None:

        """Run preprocessing to generate TGN data from events."""

        try:
            # Import here to avoid circular dependencies
            from data_pipeline.processors.preprocessing import build_tgn
            
            src_path: Path = events_file
            try:
                fd, tmp_name = tempfile.mkstemp(suffix=".jsonl")
                os.close(fd)  # we will reopen with text mode
                tmp_path = Path(tmp_name)

                with open(events_file, "r", encoding="utf-8") as fin, \
                    open(tmp_path, "w", encoding="utf-8") as fout:
                    for line in fin:
                        if not line.strip():
                            continue
                        obj = json.loads(line)

                        # If missing, derive a user_id from other plausible fields or fallback
                        if "user_id" not in obj:
                            obj["user_id"] = (
                                obj.get("actor_id")
                                or obj.get("author_id")
                                or obj.get("u_id")
                                or obj.get("channel_id")
                                or obj.get("account_id")
                                or "unknown"
                            )

                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

                src_path = tmp_path
                logger.info(f"Normalised events for preprocessing → {src_path}")

            except Exception as norm_err:
                logger.warning(f"Could not normalise events ({norm_err}); using original file")
                src_path = events_file  # fall back

            
            output_file = self.datasets_dir / "tgn_edges_basic.npz"
            logger.info(f"Running preprocessing on {src_path}")
            build_tgn(
                events_path=str(src_path),
                output_path=str(output_file),
                force=True,  # Always rebuild for training
            )
            logger.info(f"Preprocessing completed: {output_file}")

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
        finally:
            
            try:
                if 'tmp_path' in locals() and tmp_path.exists():
                    if tmp_path.resolve() != events_file.resolve(): # Only delete if it isn't the original file we were given
                        tmp_path.unlink()
            except Exception:
                pass
    
       

    def force_training(self) -> None:
        """Force immediate training regardless of schedule."""
        logger.info("Forcing immediate model training...")
        self._trigger_training()


class HourlyDataCollector:
    """Collects and stores hourly snapshots of event data."""
    
    def __init__(self, database: EventDatabase):
        """Initialize hourly data collector.
        
        Args:
            database: Event database instance
        """
        self.database = database
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info("Hourly data collector initialized")
    
    def start(self) -> None:
        """Start hourly data collection."""
        if self.is_running:
            logger.warning("Hourly collector is already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("Hourly data collector started")
    
    def stop(self) -> None:
        """Stop hourly data collection."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Hourly data collector stopped")
    
    def _collection_loop(self) -> None:
        """Main loop for hourly data collection."""
        while self.is_running:
            try:
                # Wait until the next hour
                now = utc_now()
                next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                wait_seconds = (next_hour - now).total_seconds()
                
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                
                if self.is_running:
                    self._collect_hourly_snapshot()
                
            except Exception as e:
                logger.error(f"Error in hourly collection: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _collect_hourly_snapshot(self) -> None:
        """Collect hourly data snapshot."""
        try:
            now = utc_now()
            hour_start = now.replace(minute=0, second=0, microsecond=0)
            prev_hour = hour_start - timedelta(hours=1)
            
            # Get events from the last hour
            events = self.database.get_events_since(prev_hour)
            
            # Log hourly statistics
            sources = {}
            for event in events:
                source = event.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            logger.info(f"Hourly snapshot: {len(events)} total events")
            for source, count in sources.items():
                logger.info(f"  {source}: {count} events")
            
        except Exception as e:
            logger.error(f"Failed to collect hourly snapshot: {e}")