"""Base collector class for data collection sources."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from utils.logging import get_logger
from utils.datetime import now_iso

logger = get_logger(__name__)


class BaseCollector(ABC):
    """Abstract base class for all data collectors.
    
    Provides common functionality for event processing, logging,
    and error handling across different data sources.
    """
    
    def __init__(
        self, 
        source_name: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize the base collector.
        
        Args:
            source_name: Name of the data source (e.g., "twitter", "youtube")
            on_event: Callback function to handle processed events
        """
        self.source_name = source_name
        self.on_event = on_event or self._default_event_handler
        self.is_running = False
        
    def _default_event_handler(self, event: Dict[str, Any]) -> None:
        """Default event handler that logs the event."""
        logger.info(f"[{self.source_name}] Event: {event}")
        
    def _create_base_event(self, **kwargs) -> Dict[str, Any]:
        """Create a base event structure with common fields.
        
        Args:
            **kwargs: Additional fields to include in the event
            
        Returns:
            Event dictionary with base structure
        """
        base_event = {
            "timestamp": now_iso(),
            "source": self.source_name,
        }
        base_event.update(kwargs)
        return base_event
        
    def _emit_event(self, event: Dict[str, Any]) -> None:
        """Safely emit an event through the callback.
        
        Args:
            event: Event dictionary to emit
        """
        try:
            self.on_event(event)
        except Exception as e:
            logger.error(f"[{self.source_name}] Error in event callback: {e}")
            
    def _handle_error(self, error: Exception, context: str = "") -> None:
        """Handle errors consistently across collectors.
        
        Args:
            error: The exception that occurred
            context: Optional context about when the error occurred
        """
        context_str = f" ({context})" if context else ""
        logger.error(f"[{self.source_name}] Error{context_str}: {error}")
        
    @abstractmethod
    def start(self) -> None:
        """Start the data collection process."""
        pass
        
    @abstractmethod
    def stop(self) -> None:
        """Stop the data collection process."""
        pass


class SimulatedCollector(BaseCollector):
    """Base class for simulated/fake data collectors."""
    
    def simulate_events(
        self, 
        n_events: int = 10, 
        delay: float = 1.0,
        event_generator: Optional[Callable[[int], Dict[str, Any]]] = None
    ) -> None:
        """Generate and emit simulated events.
        
        Args:
            n_events: Number of events to generate
            delay: Delay between events in seconds
            event_generator: Custom function to generate event data
        """
        import time
        
        logger.info(f"[{self.source_name}] Starting simulation with {n_events} events")
        self.is_running = True
        
        try:
            for i in range(n_events):
                if not self.is_running:
                    break
                    
                if event_generator:
                    event_data = event_generator(i)
                else:
                    event_data = self._generate_default_event(i)
                    
                event = self._create_base_event(**event_data)
                self._emit_event(event)
                
                if i < n_events - 1:  # Don't sleep after the last event
                    time.sleep(delay)
                    
        except Exception as e:
            self._handle_error(e, "simulation")
        finally:
            self.is_running = False
            logger.info(f"[{self.source_name}] Simulation completed")
            
    def _generate_default_event(self, index: int) -> Dict[str, Any]:
        """Generate a default simulated event.
        
        Args:
            index: Event index/counter
            
        Returns:
            Event data dictionary
        """
        return {
            "event_id": f"{self.source_name}_{index}",
            "content": f"Simulated {self.source_name} event #{index}",
            "index": index
        }
        
    def start(self) -> None:
        """Start simulation with default parameters."""
        self.simulate_events()
        
    def stop(self) -> None:
        """Stop the simulation."""
        self.is_running = False