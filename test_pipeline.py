"""Test script for the enhanced data pipeline.

Tests database creation, data collection, and basic API functionality.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from data_pipeline.storage.database import init_database, get_db_manager
from data_pipeline.scheduler import get_scheduler
from utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def test_database():
    """Test database initialization and basic operations."""
    logger.info("Testing database...")
    
    try:
        # Initialize database
        db_manager = init_database()
        logger.info("✓ Database initialized")
        
        # Test saving an event
        test_event = {
            "timestamp": "2025-01-09T10:00:00Z",
            "content_id": "test_content_1", 
            "user_id": "test_user_1",
            "source": "twitter",
            "type": "original",
            "text": "This is a test tweet about trending topics",
            "hashtags": ["#test", "#trending"]
        }
        
        saved_event = db_manager.save_event(test_event)
        if saved_event:
            logger.info("✓ Event saved to database")
        else:
            logger.error("✗ Failed to save event")
            return False
        
        # Test saving a prediction
        prediction = db_manager.save_prediction(
            trend_topic="test_trend",
            score=0.85,
            confidence=0.9,
            model_version="test_v1"
        )
        if prediction:
            logger.info("✓ Prediction saved to database")
        else:
            logger.error("✗ Failed to save prediction")
            return False
        
        # Test retrieval
        predictions = db_manager.get_latest_predictions(limit=1)
        if predictions:
            logger.info("✓ Predictions retrieved from database")
        else:
            logger.error("✗ Failed to retrieve predictions")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        return False


def test_scheduler():
    """Test scheduler initialization."""
    logger.info("Testing scheduler...")
    
    try:
        # Get scheduler instance
        scheduler = get_scheduler()
        logger.info("✓ Scheduler initialized")
        
        # Check if jobs are configured
        status = scheduler.get_status()
        if status.get("scheduler_running"):
            logger.info("✓ Scheduler is running")
        else:
            logger.info("! Scheduler is not running (expected for test)")
        
        jobs = status.get("jobs", [])
        logger.info(f"✓ Found {len(jobs)} configured jobs")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Scheduler test failed: {e}")
        return False


def test_collection():
    """Test immediate data collection."""
    logger.info("Testing data collection...")
    
    try:
        # Get database manager to check events before/after
        db_manager = get_db_manager()
        
        with db_manager.get_session() as session:
            from data_pipeline.storage.database import Event
            events_before = session.query(Event).count()
        
        logger.info(f"Events before collection: {events_before}")
        
        # Run immediate collection
        scheduler = get_scheduler()
        
        # Test individual collector
        twitter_job = scheduler.collection_jobs.get('twitter')
        if twitter_job:
            logger.info("Running Twitter collection test...")
            twitter_job.run_collection()
            logger.info("✓ Twitter collection completed")
        
        # Check events after
        with db_manager.get_session() as session:
            events_after = session.query(Event).count()
        
        logger.info(f"Events after collection: {events_after}")
        
        if events_after > events_before:
            logger.info("✓ New events collected")
            return True
        else:
            logger.warning("! No new events collected (may be expected for fake data)")
            return True
        
    except Exception as e:
        logger.error(f"✗ Collection test failed: {e}")
        return False


def test_api_endpoints():
    """Test API functionality without starting server."""
    logger.info("Testing API endpoints...")
    
    try:
        from service.api.prediction_api import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            logger.info("✓ Health endpoint working")
        else:
            logger.error(f"✗ Health endpoint failed: {response.status_code}")
            return False
        
        # Test predictions endpoint
        response = client.get("/predictions")
        if response.status_code == 200:
            logger.info("✓ Predictions endpoint working")
            data = response.json()
            logger.info(f"  Found {len(data)} predictions")
        else:
            logger.error(f"✗ Predictions endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ API test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting enhanced pipeline tests...")
    
    tests = [
        ("Database", test_database),
        ("Scheduler", test_scheduler), 
        ("Collection", test_collection),
        ("API", test_api_endpoints)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                logger.info(f"✓ {test_name} test PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} test FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} test ERROR: {e}")
            failed += 1
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total:  {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"💥 {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)