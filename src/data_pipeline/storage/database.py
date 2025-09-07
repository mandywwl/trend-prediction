"""Simple database storage for event data using SQLite."""

import sqlite3
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from utils.logging import get_logger

logger = get_logger(__name__)


class EventDatabase:
    """Simple SQLite database for storing streaming events."""
    
    def __init__(self, db_path: Path):
        """Initialize database connection and create tables."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Initialize database schema
        self._init_schema()
        logger.info(f"Event database initialized at {self.db_path}")
    
    def _init_schema(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    type TEXT NOT NULL,
                    user_id TEXT,
                    content_id TEXT,
                    text TEXT,
                    hashtags TEXT,  -- JSON array
                    metadata TEXT,  -- JSON object for other fields
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON events(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON events(created_at)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper locking."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                yield conn
            finally:
                conn.close()
    
    def store_event(self, event: Dict[str, Any]) -> None:
        """Store a single event in the database."""
        try:
            with self._get_connection() as conn:
                # Extract main fields
                timestamp = event.get('timestamp', '')
                source = event.get('source', '')
                event_type = event.get('type', '')
                user_id = event.get('user_id')
                content_id = event.get('content_id')
                text = event.get('text', '')
                
                # Handle hashtags
                hashtags = event.get('hashtags', [])
                hashtags_json = json.dumps(hashtags) if hashtags else None
                
                # Store other fields as metadata
                metadata_fields = {k: v for k, v in event.items() 
                                 if k not in {'timestamp', 'source', 'type', 'user_id', 
                                            'content_id', 'text', 'hashtags'}}
                metadata_json = json.dumps(metadata_fields) if metadata_fields else None
                
                conn.execute("""
                    INSERT INTO events 
                    (timestamp, source, type, user_id, content_id, text, hashtags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, source, event_type, user_id, content_id, text, 
                      hashtags_json, metadata_json))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
    
    def get_events_since(self, since: datetime) -> List[Dict[str, Any]]:
        """Get all events since a given timestamp."""
        since_iso = since.isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT timestamp, source, type, user_id, content_id, text, hashtags, metadata
                FROM events 
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (since_iso,))
            
            events = []
            for row in cursor:
                event = {
                    'timestamp': row[0],
                    'source': row[1],
                    'type': row[2],
                    'user_id': row[3],
                    'content_id': row[4],
                    'text': row[5],
                }
                
                # Parse hashtags
                if row[6]:
                    event['hashtags'] = json.loads(row[6])
                
                # Parse metadata
                if row[7]:
                    metadata = json.loads(row[7])
                    event.update(metadata)
                
                events.append(event)
            
            return events
    
    def get_event_count(self) -> int:
        """Get total number of events in database."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]
    
    def export_to_jsonl(self, output_path: Path, since: Optional[datetime] = None) -> int:
        """Export events to JSONL format for preprocessing."""
        if since:
            events = self.get_events_since(since)
        else:
            events = self.get_all_events()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for event in events:
                json.dump(event, f)
                f.write('\n')
        
        logger.info(f"Exported {len(events)} events to {output_path}")
        return len(events)
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events from database."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT timestamp, source, type, user_id, content_id, text, hashtags, metadata
                FROM events 
                ORDER BY timestamp
            """)
            
            events = []
            for row in cursor:
                event = {
                    'timestamp': row[0],
                    'source': row[1],
                    'type': row[2],
                    'user_id': row[3],
                    'content_id': row[4],
                    'text': row[5],
                }
                
                # Parse hashtags
                if row[6]:
                    event['hashtags'] = json.loads(row[6])
                
                # Parse metadata
                if row[7]:
                    metadata = json.loads(row[7])
                    event.update(metadata)
                
                events.append(event)
            
            return events