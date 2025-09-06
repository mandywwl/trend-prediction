"""API service for trend predictions dashboard.

Provides REST API endpoints for:
- Live trend predictions
- Collection status monitoring  
- Training metrics
- Health checks
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from data_pipeline.storage.database import get_db_manager, TrendPrediction, Event, ModelTraining
from data_pipeline.scheduler import get_scheduler
from utils.logging import get_logger

logger = get_logger(__name__)


# Pydantic models for API responses
class TrendPredictionResponse(BaseModel):
    """Response model for trend predictions."""
    trend_topic: str
    score: float
    confidence: Optional[float]
    timestamp: str
    source_events_count: Optional[int]
    model_version: Optional[str]


class CollectionStatusResponse(BaseModel):
    """Response model for collection status."""
    source: str
    last_collection_time: str
    events_collected: int
    status: str
    error_message: Optional[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    database_connected: bool
    scheduler_running: bool
    total_events: int
    latest_prediction_time: Optional[str]


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    id: int
    started_at: str
    completed_at: Optional[str]
    status: str
    model_version: Optional[str]
    events_count: Optional[int]
    metrics: Optional[Dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting API service...")
    yield
    logger.info("Shutting down API service...")


# Create FastAPI app
app = FastAPI(
    title="Trend Prediction API",
    description="API for accessing trend predictions and monitoring data pipeline",
    version="1.0.0",
    lifespan=lifespan
)


def get_db():
    """Dependency to get database manager."""
    return get_db_manager()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Trend Prediction API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(db_manager=Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database connection
        database_connected = True
        total_events = 0
        latest_prediction_time = None
        
        try:
            with db_manager.get_session() as session:
                # Count total events
                total_events = session.query(Event).count()
                
                # Get latest prediction time
                latest_prediction = session.query(TrendPrediction)\
                    .order_by(TrendPrediction.timestamp.desc()).first()
                if latest_prediction:
                    latest_prediction_time = latest_prediction.timestamp.isoformat()
                    
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            database_connected = False
        
        # Check scheduler status
        try:
            scheduler = get_scheduler()
            scheduler_running = scheduler.scheduler.running
        except Exception:
            scheduler_running = False
        
        status = "healthy" if database_connected and scheduler_running else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            database_connected=database_connected,
            scheduler_running=scheduler_running,
            total_events=total_events,
            latest_prediction_time=latest_prediction_time
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions", response_model=List[TrendPredictionResponse])
async def get_predictions(
    limit: int = 10,
    hours_back: int = 24,
    db_manager=Depends(get_db)
):
    """Get latest trend predictions."""
    try:
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)
        
        with db_manager.get_session() as session:
            predictions = session.query(TrendPrediction)\
                .filter(TrendPrediction.timestamp >= start_time)\
                .order_by(TrendPrediction.timestamp.desc())\
                .limit(limit).all()
        
        return [
            TrendPredictionResponse(
                trend_topic=pred.trend_topic,
                score=pred.score,
                confidence=pred.confidence,
                timestamp=pred.timestamp.isoformat(),
                source_events_count=pred.source_events_count,
                model_version=pred.model_version
            )
            for pred in predictions
        ]
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/top", response_model=List[TrendPredictionResponse])
async def get_top_predictions(
    limit: int = 5,
    hours_back: int = 6,
    db_manager=Depends(get_db)
):
    """Get top trending predictions."""
    try:
        # Get recent predictions and find highest scoring ones
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)
        
        with db_manager.get_session() as session:
            # Get latest prediction for each topic
            subquery = session.query(
                TrendPrediction.trend_topic,
                session.query(TrendPrediction.timestamp)\
                    .filter(TrendPrediction.trend_topic == TrendPrediction.trend_topic)\
                    .filter(TrendPrediction.timestamp >= start_time)\
                    .order_by(TrendPrediction.timestamp.desc())\
                    .limit(1).scalar_subquery().label('latest_time')
            ).filter(TrendPrediction.timestamp >= start_time)\
             .group_by(TrendPrediction.trend_topic).subquery()
            
            predictions = session.query(TrendPrediction)\
                .join(subquery, 
                      (TrendPrediction.trend_topic == subquery.c.trend_topic) &
                      (TrendPrediction.timestamp == subquery.c.latest_time))\
                .order_by(TrendPrediction.score.desc())\
                .limit(limit).all()
        
        return [
            TrendPredictionResponse(
                trend_topic=pred.trend_topic,
                score=pred.score,
                confidence=pred.confidence,
                timestamp=pred.timestamp.isoformat(),
                source_events_count=pred.source_events_count,
                model_version=pred.model_version
            )
            for pred in predictions
        ]
        
    except Exception as e:
        logger.error(f"Error getting top predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection/status", response_model=List[CollectionStatusResponse])
async def get_collection_status(db_manager=Depends(get_db)):
    """Get data collection status for all sources."""
    try:
        collection_statuses = db_manager.get_collection_status()
        
        return [
            CollectionStatusResponse(
                source=cs.source,
                last_collection_time=cs.last_collection_time.isoformat(),
                events_collected=cs.events_collected,
                status=cs.status,
                error_message=cs.error_message
            )
            for cs in collection_statuses
        ]
        
    except Exception as e:
        logger.error(f"Error getting collection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/status", response_model=List[TrainingStatusResponse])
async def get_training_status(
    limit: int = 10,
    db_manager=Depends(get_db)
):
    """Get model training status."""
    try:
        with db_manager.get_session() as session:
            training_runs = session.query(ModelTraining)\
                .order_by(ModelTraining.started_at.desc())\
                .limit(limit).all()
        
        return [
            TrainingStatusResponse(
                id=tr.id,
                started_at=tr.started_at.isoformat(),
                completed_at=tr.completed_at.isoformat() if tr.completed_at else None,
                status=tr.status,
                model_version=tr.model_version,
                events_count=tr.events_count,
                metrics=tr.metrics
            )
            for tr in training_runs
        ]
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events/stats")
async def get_event_stats(
    hours_back: int = 24,
    db_manager=Depends(get_db)
):
    """Get event statistics."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)
        
        with db_manager.get_session() as session:
            # Total events in time range
            total_events = session.query(Event)\
                .filter(Event.timestamp >= start_time).count()
            
            # Events by source
            from sqlalchemy import func
            events_by_source = session.query(
                Event.source,
                func.count(Event.id).label('count')
            ).filter(Event.timestamp >= start_time)\
             .group_by(Event.source).all()
            
            # Events by hour
            events_by_hour = session.query(
                func.date_trunc('hour', Event.timestamp).label('hour'),
                func.count(Event.id).label('count')
            ).filter(Event.timestamp >= start_time)\
             .group_by(func.date_trunc('hour', Event.timestamp))\
             .order_by('hour').all()
        
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours_back
            },
            "total_events": total_events,
            "events_by_source": [
                {"source": source, "count": count}
                for source, count in events_by_source
            ],
            "events_by_hour": [
                {"hour": hour.isoformat(), "count": count}
                for hour, count in events_by_hour
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting event stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collection/trigger")
async def trigger_collection(
    background_tasks: BackgroundTasks,
    source: Optional[str] = None
):
    """Trigger immediate data collection."""
    try:
        scheduler = get_scheduler()
        
        def run_collection():
            if source:
                # Run collection for specific source
                if source in scheduler.collection_jobs:
                    scheduler.collection_jobs[source].run_collection()
                else:
                    raise ValueError(f"Unknown source: {source}")
            else:
                # Run collection for all sources
                scheduler.run_immediate_collection()
        
        background_tasks.add_task(run_collection)
        
        return {
            "message": f"Collection triggered for {source or 'all sources'}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictions/generate")
async def trigger_prediction_generation(background_tasks: BackgroundTasks):
    """Trigger immediate prediction generation."""
    try:
        scheduler = get_scheduler()
        
        background_tasks.add_task(scheduler.prediction_job.generate_predictions)
        
        return {
            "message": "Prediction generation triggered",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering prediction generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status and job information."""
    try:
        scheduler = get_scheduler()
        status = scheduler.get_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Initialize database and scheduler
    from data_pipeline.storage.database import init_database
    from data_pipeline.scheduler import start_scheduler
    
    logger.info("Initializing database...")
    init_database()
    
    logger.info("Starting scheduler...")
    start_scheduler()
    
    logger.info("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)