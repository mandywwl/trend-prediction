"""Twitter/X data collector using Tweepy v2 API."""

import tweepy
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List

from .base import BaseCollector, SimulatedCollector
from utils.logging import get_logger

logger = get_logger(__name__)


class TwitterCollector(tweepy.StreamingClient, BaseCollector):
    """Twitter/X Streaming Collector using Tweepy v2.
    
    Ingests tweets in real-time and standardizes events for processing.
    """

    def __init__(
        self, 
        bearer_token: str, 
        keywords: Optional[List[str]] = None, 
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize Twitter collector.
        
        Args:
            bearer_token: Twitter/X API bearer token
            keywords: List of keywords or hashtags to track (default: all)
            on_event: Callback function to handle each processed event
        """
        tweepy.StreamingClient.__init__(self, bearer_token)
        BaseCollector.__init__(self, "twitter", on_event)
        
        self.keywords = keywords or []

        # Add stream rules if provided
        if self.keywords:
            self.add_rules(tweepy.StreamRule(" OR ".join(self.keywords)))

    def on_tweet(self, tweet):
        """Called for every new tweet event matching the stream rules."""
        try:
            # Standardize event schema
            event_data = {
                "user_id": f"u{tweet.author_id}",
                "content_id": f"t{tweet.id}",
                "hashtags": (
                    [tag["tag"].lower() for tag in tweet.entities.get("hashtags", [])]
                    if tweet.entities
                    else []
                ),
                "type": (
                    "original"
                    if not tweet.referenced_tweets
                    else tweet.referenced_tweets[0].type
                ),
                "text": tweet.text,
            }

            event = self._create_base_event(**event_data)
            self._emit_event(event)
            
            logger.info(f"Received tweet {event['content_id']} from {event['user_id']}")
            
        except Exception as e:
            self._handle_error(e, "processing tweet")

    def on_errors(self, errors):
        """Handle streaming errors."""
        self._handle_error(Exception(f"Stream errors: {errors}"), "streaming")

    def on_connection_error(self):
        """Handle connection errors."""
        logger.warning("Connection error. Restarting...")
        self.disconnect()
        
    def start(self) -> None:
        """Start the Twitter stream."""
        logger.info("Starting Twitter stream...")
        self.is_running = True
        self.filter(
            tweet_fields=["created_at", "author_id", "entities", "referenced_tweets"],
            expansions=[],
            threaded=False,
        )
        
    def stop(self) -> None:
        """Stop the Twitter stream."""
        self.is_running = False
        self.disconnect()


class FakeTwitterCollector(SimulatedCollector):
    """Simulated Twitter collector for development and testing."""
    
    def __init__(
        self, 
        keywords: Optional[List[str]] = None, 
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize fake Twitter collector.
        
        Args:
            keywords: Unused, but kept for API compatibility
            on_event: Callback function to handle events
        """
        super().__init__("twitter", on_event)
        self.keywords = keywords or []
        
    def _generate_default_event(self, index: int) -> Dict[str, Any]:
        """Generate a simulated Twitter event."""
        return {
            "user_id": f"u{index % 5}",  # 5 fake users
            "content_id": f"t{index}",
            "hashtags": ["test", "mock"] if index % 2 == 0 else ["fyp", "viral"],
            "type": "original",
            "text": f"This is a simulated tweet event number {index}.",
        }


def start_twitter_stream(
    bearer_token: str, 
    keywords: Optional[List[str]] = None, 
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
) -> TwitterCollector:
    """Entry point for launching the Twitter stream collector.
    
    Args:
        bearer_token: Twitter API bearer token
        keywords: Keywords to track
        on_event: Event callback function
        
    Returns:
        TwitterCollector instance
    """
    collector = TwitterCollector(
        bearer_token=bearer_token, 
        keywords=keywords, 
        on_event=on_event
    )
    collector.start()
    return collector


def fake_twitter_stream(
    keywords: Optional[List[str]] = None, 
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None, 
    n_events: int = 10, 
    delay: float = 1.0
) -> None:
    """Simulate streaming Twitter events for development without API access.
    
    Args:
        keywords: Unused, but kept for API compatibility
        on_event: Callback function to process each event
        n_events: Number of fake events to emit
        delay: Seconds between each event
    """
    collector = FakeTwitterCollector(keywords=keywords, on_event=on_event)
    collector.simulate_events(n_events=n_events, delay=delay)
