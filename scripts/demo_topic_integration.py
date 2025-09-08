#!/usr/bin/env python3
"""
Demonstration of how to integrate the topic labeling pipeline with RuntimeGlue.

This script shows how the topic labeling pipeline can be called periodically
to update topic labels using the existing RuntimeGlue._update_topic_lookup method.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from generate_topic_labels import generate_topic_labels
try:
    from utils.path_utils import find_repo_root
    from service.runtime_glue import RuntimeGlue, RuntimeConfig
except ImportError as e:
    print(f"Could not import RuntimeGlue components: {e}")
    print("This is expected in the sandbox environment")
    RuntimeGlue = None


def update_topic_labels_via_runtime_glue():
    """Demonstrate updating topic labels using RuntimeGlue._update_topic_lookup."""
    
    if RuntimeGlue is None:
        print("RuntimeGlue not available - using direct file update method")
        return run_standalone_update()
    
    try:
        # Get repository paths
        repo_root = find_repo_root()
        events_path = repo_root / "datasets" / "events.jsonl"
        topic_lookup_path = repo_root / "datasets" / "topic_lookup.json"
        
        # Create RuntimeConfig 
        config = RuntimeConfig(
            predictions_cache_path=str(repo_root / "datasets" / "predictions_cache.json"),
            metrics_snapshot_dir=str(repo_root / "datasets" / "metrics_hourly"),
            topic_lookup_path=str(topic_lookup_path),
        )
        
        # Initialize RuntimeGlue (with dummy handler for this demo)
        class DummyHandler:
            def on_event(self, event):
                return {}
        
        glue = RuntimeGlue(event_handler=DummyHandler(), config=config)
        
        # Generate new topic labels
        print("Generating topic labels...")
        updated_lookup = generate_topic_labels(
            events_path=events_path,
            topic_lookup_path=topic_lookup_path,
            n_clusters=8,
            max_events=400
        )
        
        # Update topics using RuntimeGlue method
        print("Updating topics via RuntimeGlue._update_topic_lookup...")
        for topic_id, label in updated_lookup.items():
            # Convert to integer for the RuntimeGlue method
            try:
                topic_id_int = int(topic_id)
                glue._update_topic_lookup(topic_id_int, label)
            except ValueError:
                # Skip non-integer topic IDs
                continue
        
        print("Topic labels updated successfully via RuntimeGlue!")
        
    except Exception as e:
        print(f"Error using RuntimeGlue: {e}")
        print("Falling back to standalone update...")
        return run_standalone_update()


def run_standalone_update():
    """Run the topic labeling pipeline standalone (current implementation)."""
    
    try:
        repo_root = Path(__file__).parent.parent
        events_path = repo_root / "datasets" / "events.jsonl"
        topic_lookup_path = repo_root / "datasets" / "topic_lookup.json"
        
        if not events_path.exists():
            print(f"Error: Events file not found at {events_path}")
            return False
        
        print("Running standalone topic labeling pipeline...")
        updated_lookup = generate_topic_labels(
            events_path=events_path,
            topic_lookup_path=topic_lookup_path,
            n_clusters=8,
            max_events=400
        )
        
        if updated_lookup:
            print(f"Successfully updated {len(updated_lookup)} topics")
            
            # Show some sample results
            print("\nSample updated topics:")
            for i, (topic_id, label) in enumerate(list(updated_lookup.items())[:5]):
                print(f"  {topic_id}: {label}")
            
            return True
        else:
            print("No topics were updated")
            return False
            
    except Exception as e:
        print(f"Error in standalone topic labeling: {e}")
        return False


def main():
    """Main demonstration function."""
    
    print("Topic Labeling Pipeline Integration Demo")
    print("=" * 50)
    
    # First try with RuntimeGlue integration
    print("Attempting integration with RuntimeGlue...")
    update_topic_labels_via_runtime_glue()
    
    print("\nDemonstration complete!")
    print("\nHow to use this in production:")
    print("1. Call generate_topic_labels() periodically (e.g., daily)")
    print("2. Use RuntimeGlue._update_topic_lookup() to persist changes")
    print("3. The dashboard will automatically show updated labels")


if __name__ == "__main__":
    main()