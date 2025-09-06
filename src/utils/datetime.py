"""DateTime utilities for the project."""

from datetime import datetime, timezone
from typing import Union
import time


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Get current timestamp in ISO format."""
    return utc_now().isoformat()


def parse_iso_timestamp(ts_iso: str) -> datetime:
    """Parse ISO timestamp string to datetime object."""
    return datetime.fromisoformat(ts_iso.replace('Z', '+00:00'))


def timestamp_to_seconds(ts_iso: str) -> float:
    """Convert ISO timestamp to seconds since epoch."""
    dt = parse_iso_timestamp(ts_iso)
    return dt.timestamp()


def seconds_to_iso(seconds: float) -> str:
    """Convert seconds since epoch to ISO timestamp."""
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.isoformat()


def time_since_epoch() -> float:
    """Get current time as seconds since epoch.""" 
    return time.time()