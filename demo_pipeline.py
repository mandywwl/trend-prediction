#!/usr/bin/env python3
"""
Demo script for the Enhanced Data Pipeline

This script demonstrates the new pipeline capabilities:
1. Database storage for events and predictions
2. Data collection simulation
3. Prediction generation
4. Dashboard-ready API endpoints

Usage:
    python demo_pipeline.py [--mode collect|predict|api|status]
"""

import os
import sys
import time
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

# Import pipeline components
from data_pipeline.storage.database import init_database, get_db_manager
from utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def simulate_data_collection():
    """Simulate hourly data collection from all platforms."""
    logger.info("🔄 Simulating hourly data collection...")
    
    db_manager = get_db_manager()
    
    # Simulate Twitter events
    twitter_events = [
        {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'content_id': f'tweet_{int(time.time())}_{i}',
            'user_id': f'user_{i % 10}',
            'source': 'twitter',
            'type': 'original' if i % 3 == 0 else 'retweet',
            'text': f'Gen Z trend #{i}: This is so viral! #fyp #trending',
            'hashtags': ['#fyp', '#trending', '#GenZ']
        }
        for i in range(10)
    ]
    
    # Simulate YouTube events
    youtube_events = [
        {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'content_id': f'video_{int(time.time())}_{i}',
            'user_id': f'creator_{i % 5}',
            'source': 'youtube',
            'type': 'upload',
            'text': f'Viral dance trend #{i} taking over social media',
            'hashtags': ['#dance', '#viral', '#youtube']
        }
        for i in range(5)
    ]
    
    # Simulate Google Trends events
    trends_events = [
        {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'content_id': f'trend_{int(time.time())}_{i}',
            'source': 'google_trends',
            'type': 'trend',
            'text': trend,
            'context': ['Technology', 'Social Media']
        }
        for i, trend in enumerate(['AI chatbots', 'Social media trends', 'Gen Z behavior', 'Viral content'])
    ]
    
    # Save all events
    total_saved = 0
    for events, source in [(twitter_events, 'Twitter'), (youtube_events, 'YouTube'), (trends_events, 'Google Trends')]:
        for event in events:
            saved = db_manager.save_event(event)
            if saved:
                total_saved += 1
        logger.info(f"✓ Saved {len(events)} {source} events")
    
    # Update collection status
    for source in ['twitter', 'youtube', 'google_trends']:
        event_count = len([e for e in twitter_events + youtube_events + trends_events if e['source'] == source])
        db_manager.update_collection_status(
            source=source,
            events_collected=event_count,
            status="active"
        )
    
    logger.info(f"✅ Collection completed: {total_saved} events saved to database")
    return total_saved


def generate_predictions():
    """Generate trend predictions from recent events."""
    logger.info("🤖 Generating trend predictions...")
    
    db_manager = get_db_manager()
    
    # Get recent events for prediction
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=6)
    
    recent_events = db_manager.get_events_for_training(start_time, end_time)
    logger.info(f"📊 Analyzing {len(recent_events)} recent events")
    
    # Generate predictions based on event patterns
    predictions = {}
    
    if recent_events:
        # Analyze hashtags
        hashtag_counts = {}
        text_keywords = {}
        
        for event in recent_events:
            # Count hashtags
            if event.hashtags:
                for tag in event.hashtags:
                    hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
            
            # Extract keywords from text
            if event.text:
                words = event.text.lower().split()
                for word in words:
                    if len(word) > 3:  # Only consider longer words
                        text_keywords[word] = text_keywords.get(word, 0) + 1
        
        # Generate predictions based on patterns
        if hashtag_counts:
            top_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for tag, count in top_hashtags:
                score = min(0.9, count / len(recent_events) + 0.3)
                predictions[f"hashtag_trend_{tag.replace('#', '')}"] = score
        
        if text_keywords:
            top_keywords = sorted(text_keywords.items(), key=lambda x: x[1], reverse=True)[:3]
            for keyword, count in top_keywords:
                score = min(0.85, count / len(recent_events) + 0.2)
                predictions[f"keyword_trend_{keyword}"] = score
        
        # Add some Gen Z specific predictions
        gen_z_indicators = ['viral', 'fyp', 'trending', 'tiktok', 'gen', 'z']
        gen_z_score = 0.0
        
        for event in recent_events:
            if event.text:
                text_lower = event.text.lower()
                for indicator in gen_z_indicators:
                    if indicator in text_lower:
                        gen_z_score += 0.1
        
        if gen_z_score > 0:
            predictions["gen_z_engagement"] = min(0.95, gen_z_score)
    
    # Default predictions if no recent events
    if not predictions:
        predictions = {
            "emerging_trends": 0.65,
            "social_media_buzz": 0.58,
            "gen_z_culture": 0.72
        }
    
    # Save predictions to database
    saved_count = 0
    for topic, score in predictions.items():
        saved = db_manager.save_prediction(
            trend_topic=topic,
            score=score,
            confidence=0.8,
            source_events_count=len(recent_events),
            model_version="demo_v1",
            prediction_metadata={
                "generation_time": datetime.now(timezone.utc).isoformat(),
                "method": "pattern_analysis"
            }
        )
        if saved:
            saved_count += 1
            logger.info(f"📈 {topic}: {score:.3f}")
    
    logger.info(f"✅ Generated {saved_count} predictions")
    return saved_count


def show_status():
    """Show current pipeline status."""
    logger.info("📊 Pipeline Status Report")
    
    db_manager = get_db_manager()
    
    # Database stats
    with db_manager.get_session() as session:
        from data_pipeline.storage.database import Event, TrendPrediction, CollectionStatus
        
        total_events = session.query(Event).count()
        total_predictions = session.query(TrendPrediction).count()
        
        logger.info(f"💾 Database: {total_events} events, {total_predictions} predictions")
        
        # Events by source
        from sqlalchemy import func
        events_by_source = session.query(
            Event.source,
            func.count(Event.id).label('count')
        ).group_by(Event.source).all()
        
        logger.info("📈 Events by source:")
        for source, count in events_by_source:
            logger.info(f"  - {source}: {count} events")
        
        # Recent predictions
        recent_predictions = db_manager.get_latest_predictions(limit=5)
        logger.info(f"🎯 Latest predictions:")
        for pred in recent_predictions:
            logger.info(f"  - {pred.trend_topic}: {pred.score:.3f} (confidence: {pred.confidence or 'N/A'})")
        
        # Collection status
        collection_statuses = db_manager.get_collection_status()
        logger.info("🔄 Collection status:")
        for cs in collection_statuses:
            logger.info(f"  - {cs.source}: {cs.status} (last: {cs.last_collection_time})")


def start_api_demo():
    """Start API server for dashboard integration."""
    logger.info("🚀 Starting API server for dashboard...")
    
    try:
        from service.api.prediction_api import app
        import uvicorn
        
        logger.info("📡 API endpoints available:")
        logger.info("  - GET /health - Health check")
        logger.info("  - GET /predictions - Latest predictions")
        logger.info("  - GET /predictions/top - Top trending predictions")
        logger.info("  - GET /collection/status - Collection status")
        logger.info("  - GET /events/stats - Event statistics")
        logger.info("  - POST /collection/trigger - Trigger collection")
        logger.info("  - POST /predictions/generate - Generate predictions")
        
        logger.info("🌐 Starting server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except ImportError as e:
        logger.error(f"❌ API dependencies not available: {e}")
        logger.info("💡 Install FastAPI and Uvicorn to use API mode")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Data Pipeline Demo")
    parser.add_argument("--mode", choices=["collect", "predict", "status", "api", "full"],
                       default="full", help="Demo mode to run")
    
    args = parser.parse_args()
    
    logger.info("🎯 Enhanced Trend Prediction Pipeline Demo")
    logger.info(f"📋 Mode: {args.mode}")
    
    # Initialize database
    logger.info("🔧 Initializing database...")
    init_database()
    logger.info("✅ Database ready")
    
    if args.mode in ["collect", "full"]:
        simulate_data_collection()
        time.sleep(1)
    
    if args.mode in ["predict", "full"]:
        generate_predictions()
        time.sleep(1)
    
    if args.mode in ["status", "full"]:
        show_status()
    
    if args.mode == "api":
        start_api_demo()
    
    if args.mode == "full":
        logger.info("🎉 Demo completed! Try different modes:")
        logger.info("  python demo_pipeline.py --mode collect")
        logger.info("  python demo_pipeline.py --mode predict") 
        logger.info("  python demo_pipeline.py --mode status")
        logger.info("  python demo_pipeline.py --mode api")


if __name__ == "__main__":
    main()