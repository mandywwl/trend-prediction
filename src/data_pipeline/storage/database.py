"""Database layer for trend prediction pipeline.

Provides SQLAlchemy models and database management for storing events,
trends, and predictions with support for hourly collection and weekly training.
"""

import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Text, Float, JSON,
    Boolean, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Event(Base):
    """Table for storing raw events from data collectors."""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    content_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), index=True)
    source = Column(String(50), nullable=False, index=True)  # twitter, youtube, google_trends
    event_type = Column(String(50), nullable=False)  # original, retweet, upload, trend
    text = Column(Text)
    hashtags = Column(JSON)  # List of hashtags
    context = Column(JSON)  # Additional context data
    features = Column(JSON)  # Processed features (embeddings, etc.)
    raw_data = Column(JSON)  # Raw event data for debugging
    processed_at = Column(DateTime(timezone=True), default=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_events_timestamp_source', 'timestamp', 'source'),
        Index('idx_events_hourly', 'timestamp', 'source', 'event_type'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'content_id': self.content_id,
            'user_id': self.user_id,
            'source': self.source,
            'type': self.event_type,
            'text': self.text,
            'hashtags': self.hashtags,
            'context': self.context,
            'features': self.features,
            'raw_data': self.raw_data,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }


class TrendPrediction(Base):
    """Table for storing trend predictions and scores."""
    __tablename__ = 'trend_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    trend_topic = Column(String(255), nullable=False, index=True)
    score = Column(Float, nullable=False)
    confidence = Column(Float)
    source_events_count = Column(Integer)
    model_version = Column(String(50))
    prediction_metadata = Column(JSON)  # Additional prediction metadata (renamed from metadata)
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        Index('idx_predictions_timestamp_topic', 'timestamp', 'trend_topic'),
    )


class ModelTraining(Base):
    """Table for tracking model training runs."""
    __tablename__ = 'model_training'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    status = Column(String(50), nullable=False)  # running, completed, failed
    model_version = Column(String(50))
    training_data_start = Column(DateTime(timezone=True))
    training_data_end = Column(DateTime(timezone=True))
    events_count = Column(Integer)
    metrics = Column(JSON)  # Training metrics
    error_log = Column(Text)
    config = Column(JSON)  # Training configuration
    
    __table_args__ = (
        Index('idx_training_status_started', 'status', 'started_at'),
    )


class CollectionStatus(Base):
    """Table for tracking data collection status."""
    __tablename__ = 'collection_status'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False, index=True)
    last_collection_time = Column(DateTime(timezone=True), nullable=False)
    events_collected = Column(Integer, default=0)
    status = Column(String(50), nullable=False)  # active, error, stopped
    error_message = Column(Text)
    status_metadata = Column(JSON)  # Renamed from metadata
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_collection_source_time', 'source', 'last_collection_time'),
    )


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL. If None, uses SQLite in datasets directory.
        """
        self.logger = get_logger(f"{__name__}.DatabaseManager")
        
        if database_url is None:
            # Default to SQLite in project datasets directory
            project_root = Path(__file__).resolve().parents[3]
            db_path = project_root / "datasets" / "trend_prediction.db"
            db_path.parent.mkdir(exist_ok=True)
            database_url = f"sqlite:///{db_path}"
        
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self.logger.info(f"Database initialized: {database_url}")
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def save_event(self, event_data: Dict[str, Any]) -> Optional[Event]:
        """Save an event to the database.
        
        Args:
            event_data: Event data dictionary from collectors
            
        Returns:
            Saved Event object or None if error
        """
        try:
            with self.get_session() as session:
                # Parse timestamp
                timestamp_str = event_data.get('timestamp')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now(timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)
                
                # Create event record
                event = Event(
                    timestamp=timestamp,
                    content_id=event_data.get('content_id', ''),
                    user_id=event_data.get('user_id'),
                    source=event_data.get('source', 'unknown'),
                    event_type=event_data.get('type', 'unknown'),
                    text=event_data.get('text'),
                    hashtags=event_data.get('hashtags'),
                    context=event_data.get('context'),
                    features=event_data.get('features'),
                    raw_data=event_data
                )
                
                session.add(event)
                session.commit()
                session.refresh(event)
                
                return event
                
        except Exception as e:
            self.logger.error(f"Error saving event: {e}")
            return None
    
    def save_prediction(self, trend_topic: str, score: float, 
                       confidence: float = None, source_events_count: int = None,
                       model_version: str = None, prediction_metadata: Dict = None) -> Optional[TrendPrediction]:
        """Save a trend prediction to the database."""
        try:
            with self.get_session() as session:
                prediction = TrendPrediction(
                    timestamp=datetime.now(timezone.utc),
                    trend_topic=trend_topic,
                    score=score,
                    confidence=confidence,
                    source_events_count=source_events_count,
                    model_version=model_version,
                    prediction_metadata=prediction_metadata or {}
                )
                
                session.add(prediction)
                session.commit()
                session.refresh(prediction)
                
                return prediction
                
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            return None
    
    def get_events_for_training(self, start_time: datetime, end_time: datetime,
                               limit: int = None) -> List[Event]:
        """Get events for model training in a time range."""
        try:
            with self.get_session() as session:
                query = session.query(Event).filter(
                    Event.timestamp >= start_time,
                    Event.timestamp <= end_time
                ).order_by(Event.timestamp)
                
                if limit:
                    query = query.limit(limit)
                
                return query.all()
                
        except Exception as e:
            self.logger.error(f"Error getting training events: {e}")
            return []
    
    def get_latest_predictions(self, limit: int = 10) -> List[TrendPrediction]:
        """Get latest trend predictions."""
        try:
            with self.get_session() as session:
                return session.query(TrendPrediction)\
                    .order_by(TrendPrediction.timestamp.desc())\
                    .limit(limit).all()
                    
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return []
    
    def update_collection_status(self, source: str, events_collected: int = 0,
                               status: str = "active", error_message: str = None):
        """Update collection status for a data source."""
        try:
            with self.get_session() as session:
                # Check if status exists
                collection_status = session.query(CollectionStatus)\
                    .filter(CollectionStatus.source == source).first()
                
                if collection_status:
                    collection_status.last_collection_time = datetime.now(timezone.utc)
                    collection_status.events_collected += events_collected
                    collection_status.status = status
                    collection_status.error_message = error_message
                else:
                    collection_status = CollectionStatus(
                        source=source,
                        last_collection_time=datetime.now(timezone.utc),
                        events_collected=events_collected,
                        status=status,
                        error_message=error_message
                    )
                    session.add(collection_status)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating collection status: {e}")
    
    def get_collection_status(self) -> List[CollectionStatus]:
        """Get current collection status for all sources."""
        try:
            with self.get_session() as session:
                return session.query(CollectionStatus).all()
                
        except Exception as e:
            self.logger.error(f"Error getting collection status: {e}")
            return []


# Global database manager instance
_db_manager = None


def get_db_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
        _db_manager.create_tables()
    return _db_manager


def init_database(database_url: Optional[str] = None) -> DatabaseManager:
    """Initialize the database and return manager."""
    return get_db_manager(database_url)