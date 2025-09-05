"""Monitoring utilities for tracking system metrics."""

import time
from typing import Dict, List
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class MetricPoint:
    """Single metric measurement."""
    timestamp: float
    value: float
    tags: Dict[str, str] = None

class SimpleMetricsCollector:
    """Simple in-memory metrics collector."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
    
    def record(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        self.metrics[metric_name].append(point)
    
    def get_latest(self, metric_name: str) -> MetricPoint:
        """Get the latest value for a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_average(self, metric_name: str, window_seconds: float = 60.0) -> float:
        """Get average value over a time window."""
        if metric_name not in self.metrics:
            return 0.0
        
        cutoff_time = time.time() - window_seconds
        recent_values = [
            point.value for point in self.metrics[metric_name]
            if point.timestamp >= cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else 0.0

# Global metrics collector instance
_metrics = SimpleMetricsCollector()

def record_metric(name: str, value: float, tags: Dict[str, str] = None):
    """Record a metric value to the global collector."""
    _metrics.record(name, value, tags)

def get_metric_average(name: str, window_seconds: float = 60.0) -> float:
    """Get average metric value over time window."""
    return _metrics.get_average(name, window_seconds)