#!/usr/bin/env python3
"""
Standalone script to update topic labels in topic_lookup.json.

This script can be run independently to generate meaningful topic labels
from textual examples in the events data.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_pipeline.processors.topic_labeling import run_topic_labeling_pipeline


def main():
    """Main function to run the topic labeling pipeline."""
    print("=== Topic Labeling Pipeline ===")
    print("Generating meaningful topic labels from textual examples...")
    print()
    
    try:
        # Run the pipeline
        result = run_topic_labeling_pipeline(use_embedder=False)
        
        # Count updates
        updated_count = 0
        for topic_id, label in result.items():
            if not (label.startswith("topic_") or 
                   label.startswith("test_") or 
                   label.startswith("viral_") or 
                   label.startswith("trending_")):
                updated_count += 1
        
        print()
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 Total topics: {len(result)}")
        print(f"🔄 Updated topics: {updated_count}")
        print(f"📝 Topic lookup saved to: datasets/topic_lookup.json")
        
        if updated_count > 0:
            print("\n🏷️  Sample updated labels:")
            sample_count = 0
            for topic_id, label in result.items():
                if not (label.startswith("topic_") or 
                       label.startswith("test_") or 
                       label.startswith("viral_") or 
                       label.startswith("trending_")):
                    print(f"   • Topic {topic_id}: {label}")
                    sample_count += 1
                    if sample_count >= 5:  # Show max 5 samples
                        break
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())