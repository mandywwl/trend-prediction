"""Scheduler for automated data collection and model training.

Provides scheduling infrastructure for:
- Hourly data collection from all platforms
- Weekly model retraining
- Prediction generation and caching
- Health monitoring and metrics collection
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable, List
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from data_pipeline.storage.database import get_db_manager, ModelTraining
from data_pipeline.collectors.twitter_collector import fake_twitter_stream
from data_pipeline.collectors.youtube_collector import start_youtube_api_collector  
from data_pipeline.collectors.google_trends_collector import (
    start_google_trends_collector, fake_google_trends_stream
)
from utils.logging import get_logger
from config.config import DATA_DIR

logger = get_logger(__name__)


class CollectionJob:
    """Handles scheduled data collection from a specific platform."""
    
    def __init__(self, source: str, collector_func: Callable, 
                 collector_kwargs: Dict[str, Any] = None):
        """Initialize collection job.
        
        Args:
            source: Platform source name (twitter, youtube, google_trends)
            collector_func: Function to call for data collection
            collector_kwargs: Additional arguments for collector function
        """
        self.source = source
        self.collector_func = collector_func
        self.collector_kwargs = collector_kwargs or {}
        self.logger = get_logger(f"{__name__}.{source}")
        self.db_manager = get_db_manager()
        
        # Collection state
        self.events_collected = 0
        self.last_error = None
        self.collection_start_time = None
    
    def run_collection(self):
        """Execute data collection for this platform."""
        self.logger.info(f"Starting hourly collection for {self.source}")
        self.collection_start_time = datetime.now(timezone.utc)
        self.events_collected = 0
        self.last_error = None
        
        def on_event(event):
            """Handle collected event."""
            try:
                # Save to database
                saved_event = self.db_manager.save_event(event)
                if saved_event:
                    self.events_collected += 1
                    
                # Also save to file for compatibility
                self._save_event_to_file(event)
                
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
                self.last_error = str(e)
        
        try:
            # Update collection status to running
            self.db_manager.update_collection_status(
                source=self.source,
                status="running"
            )
            
            # Run collection with event handler
            kwargs = self.collector_kwargs.copy()
            kwargs['on_event'] = on_event
            
            self.collector_func(**kwargs)
            
            # Update collection status on success
            self.db_manager.update_collection_status(
                source=self.source,
                events_collected=self.events_collected,
                status="active"
            )
            
            self.logger.info(
                f"Completed collection for {self.source}: "
                f"{self.events_collected} events collected"
            )
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Collection failed for {self.source}: {e}")
            
            # Update collection status on error
            self.db_manager.update_collection_status(
                source=self.source,
                status="error",
                error_message=self.last_error
            )
    
    def _save_event_to_file(self, event):
        """Save event to JSONL file for compatibility."""
        try:
            import json
            event_file = DATA_DIR / "events.jsonl"
            with open(event_file, "a", encoding="utf-8") as f:
                json.dump(event, f)
                f.write("\n")
        except Exception as e:
            self.logger.warning(f"Failed to save event to file: {e}")


class TrainingJob:
    """Handles scheduled model training."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.TrainingJob")
        self.db_manager = get_db_manager()
    
    def run_training(self):
        """Execute weekly model training."""
        self.logger.info("Starting weekly model training")
        
        # Create training record
        training_record = None
        try:
            with self.db_manager.get_session() as session:
                training_record = ModelTraining(
                    started_at=datetime.now(timezone.utc),
                    status="running",
                    model_version=f"v{int(time.time())}",
                    config={"weekly_training": True}
                )
                session.add(training_record)
                session.commit()
                session.refresh(training_record)
            
            # Get training data from last week
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=7)
            
            training_events = self.db_manager.get_events_for_training(
                start_time=start_time,
                end_time=end_time
            )
            
            self.logger.info(f"Found {len(training_events)} events for training")
            
            if len(training_events) < 10:  # Minimum events needed
                raise ValueError("Insufficient training data")
            
            # Prepare training data
            training_data = self._prepare_training_data(training_events)
            
            # Run actual training (placeholder for now)
            metrics = self._run_model_training(training_data)
            
            # Update training record on success
            with self.db_manager.get_session() as session:
                training_record = session.merge(training_record)
                training_record.completed_at = datetime.now(timezone.utc)
                training_record.status = "completed"
                training_record.training_data_start = start_time
                training_record.training_data_end = end_time
                training_record.events_count = len(training_events)
                training_record.metrics = metrics
                session.commit()
            
            self.logger.info("Weekly training completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Training failed: {error_msg}")
            
            # Update training record on error
            if training_record:
                try:
                    with self.db_manager.get_session() as session:
                        training_record = session.merge(training_record)
                        training_record.status = "failed"
                        training_record.error_log = error_msg
                        session.commit()
                except Exception as update_error:
                    self.logger.error(f"Failed to update training record: {update_error}")
    
    def _prepare_training_data(self, events):
        """Prepare events data for model training."""
        # Convert events to format expected by existing training pipeline
        training_data = []
        for event in events:
            event_dict = event.to_dict()
            training_data.append(event_dict)
        
        return training_data
    
    def _run_model_training(self, training_data):
        """Run the actual model training."""
        # Placeholder for actual training logic
        # This would integrate with existing model/training/train.py
        
        self.logger.info(f"Training model with {len(training_data)} events")
        
        # Simulate training metrics
        metrics = {
            "training_events": len(training_data),
            "training_duration_seconds": 30,  # Simulated
            "final_loss": 0.15,  # Simulated
            "epochs": 10  # Simulated
        }
        
        return metrics


class PredictionJob:
    """Handles prediction generation and caching."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.PredictionJob")
        self.db_manager = get_db_manager()
    
    def generate_predictions(self):
        """Generate trend predictions for dashboard."""
        self.logger.info("Generating trend predictions")
        
        try:
            # Get recent events for prediction
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=6)  # Last 6 hours
            
            recent_events = self.db_manager.get_events_for_training(
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if not recent_events:
                self.logger.warning("No recent events for prediction")
                return
            
            # Generate predictions (placeholder)
            predictions = self._generate_trend_predictions(recent_events)
            
            # Save predictions to database
            for topic, score in predictions.items():
                self.db_manager.save_prediction(
                    trend_topic=topic,
                    score=score,
                    confidence=0.8,  # Placeholder
                    source_events_count=len(recent_events),
                    model_version="current",
                    prediction_metadata={"generation_time": datetime.now(timezone.utc).isoformat()}
                )
            
            self.logger.info(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
    
    def _generate_trend_predictions(self, events):
        """Generate trend predictions from recent events."""
        # Placeholder prediction logic
        # This would integrate with actual TGN model
        
        predictions = {
            "gen_z_trends": 0.85,
            "viral_content": 0.72,
            "social_movements": 0.68,
            "tech_innovations": 0.64,
            "entertainment": 0.58
        }
        
        return predictions


class DataPipelineScheduler:
    """Main scheduler for the data pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the scheduler.
        
        Args:
            config: Configuration dictionary with scheduling options
        """
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.DataPipelineScheduler")
        self.scheduler = BackgroundScheduler()
        self.db_manager = get_db_manager()
        
        # Collection jobs
        self.collection_jobs = {}
        self.training_job = TrainingJob()
        self.prediction_job = PredictionJob()
        
        # Setup scheduler event listeners
        self.scheduler.add_listener(self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        
        self._setup_collection_jobs()
    
    def _setup_collection_jobs(self):
        """Setup collection jobs for each platform."""
        # Twitter collection
        self.collection_jobs['twitter'] = CollectionJob(
            source='twitter',
            collector_func=fake_twitter_stream,  # Use fake for demo
            collector_kwargs={
                'keywords': ["#trending", "fyp", "viral"],
                'n_events': 50,  # Collect 50 events per hour
                'delay': 1.0
            }
        )
        
        # YouTube collection
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if youtube_api_key:
            self.collection_jobs['youtube'] = CollectionJob(
                source='youtube',
                collector_func=start_youtube_api_collector,
                collector_kwargs={
                    'api_key': youtube_api_key,
                    'max_results': 20,
                    'delay': 2.0
                }
            )
        else:
            self.logger.warning("YouTube API key not found, skipping YouTube collection")
        
        # Google Trends collection  
        self.collection_jobs['google_trends'] = CollectionJob(
            source='google_trends',
            collector_func=fake_google_trends_stream,  # Use fake for demo
            collector_kwargs={
                'n_events': 30,
                'delay': 2.0
            }
        )
    
    def start(self):
        """Start the scheduler with all jobs."""
        self.logger.info("Starting data pipeline scheduler")
        
        # Schedule hourly data collection for each platform
        for source, job in self.collection_jobs.items():
            self.scheduler.add_job(
                func=job.run_collection,
                trigger=CronTrigger(minute=0),  # Every hour at minute 0
                id=f"collect_{source}",
                name=f"Hourly {source} collection",
                max_instances=1,
                replace_existing=True
            )
            self.logger.info(f"Scheduled hourly collection for {source}")
        
        # Schedule weekly model training (Sundays at 2 AM)
        self.scheduler.add_job(
            func=self.training_job.run_training,
            trigger=CronTrigger(day_of_week=0, hour=2, minute=0),
            id="weekly_training",
            name="Weekly model training",
            max_instances=1,
            replace_existing=True
        )
        self.logger.info("Scheduled weekly model training")
        
        # Schedule prediction generation (every 30 minutes)
        self.scheduler.add_job(
            func=self.prediction_job.generate_predictions,
            trigger=IntervalTrigger(minutes=30),
            id="generate_predictions",
            name="Generate trend predictions",
            max_instances=1,
            replace_existing=True
        )
        self.logger.info("Scheduled prediction generation")
        
        # Start the scheduler
        self.scheduler.start()
        self.logger.info("Data pipeline scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.logger.info("Stopping data pipeline scheduler")
        self.scheduler.shutdown(wait=True)
        self.logger.info("Data pipeline scheduler stopped")
    
    def run_immediate_collection(self):
        """Run immediate data collection for testing."""
        self.logger.info("Running immediate data collection")
        
        for source, job in self.collection_jobs.items():
            try:
                self.logger.info(f"Running immediate collection for {source}")
                job.run_collection()
            except Exception as e:
                self.logger.error(f"Immediate collection failed for {source}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler and collection status."""
        collection_statuses = self.db_manager.get_collection_status()
        
        status = {
            "scheduler_running": self.scheduler.running,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in self.scheduler.get_jobs()
            ],
            "collection_status": [
                {
                    "source": cs.source,
                    "last_collection": cs.last_collection_time.isoformat(),
                    "events_collected": cs.events_collected,
                    "status": cs.status,
                    "error_message": cs.error_message
                }
                for cs in collection_statuses
            ]
        }
        
        return status
    
    def _job_listener(self, event):
        """Listen to job execution events."""
        if event.exception:
            self.logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            self.logger.info(f"Job {event.job_id} executed successfully")


# Global scheduler instance
_scheduler = None


def get_scheduler(config: Dict[str, Any] = None) -> DataPipelineScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DataPipelineScheduler(config)
    return _scheduler


def start_scheduler(config: Dict[str, Any] = None):
    """Start the global scheduler."""
    scheduler = get_scheduler(config)
    scheduler.start()
    return scheduler


def stop_scheduler():
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None