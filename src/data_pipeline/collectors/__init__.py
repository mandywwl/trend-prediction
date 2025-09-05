"""Data collectors for various sources."""

from .twitter_collector import fake_twitter_stream
from .youtube_collector import start_youtube_api_collector
from .google_trends_collector import start_google_trends_collector

__all__ = ["fake_twitter_stream", "start_youtube_api_collector", "start_google_trends_collector"]