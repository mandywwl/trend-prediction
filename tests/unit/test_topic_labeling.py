"""Tests for topic labeling pipeline."""

import json
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import the labeling functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from generate_topic_labels import (
    collect_texts_from_events,
    extract_keywords_from_cluster,
    generate_topic_labels
)


def test_collect_texts_from_events():
    """Test text collection from events file."""
    # Create temporary events file
    events_data = [
        {
            "timestamp": "2025-09-07T13:31:52.916756+00:00",
            "source": "youtube",
            "type": "upload", 
            "user_id": "test_user",
            "content_id": "test_video",
            "text": "Test video about machine learning"
        },
        {
            "timestamp": "2025-09-07T13:31:53.080179+00:00",
            "source": "twitter",
            "type": "original",
            "user_id": "test_user2", 
            "content_id": "test_tweet",
            "text": "This is a test tweet about AI",
            "hashtags": ["ai", "test"]
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for event in events_data:
            f.write(json.dumps(event) + '\n')
        events_path = Path(f.name)
    
    try:
        # Test text collection
        texts = collect_texts_from_events(events_path, max_events=10)
        
        assert len(texts) == 2
        assert "Test video about machine learning" in texts
        assert "This is a test tweet about AI #ai #test" in texts
        
    finally:
        # Clean up
        events_path.unlink()


def test_extract_keywords_from_cluster():
    """Test keyword extraction from text clusters."""
    # Test with machine learning related texts
    ml_texts = [
        "Machine learning and artificial intelligence",
        "Deep learning neural networks",
        "AI and ML algorithms"
    ]
    
    keywords = extract_keywords_from_cluster(ml_texts)
    assert isinstance(keywords, str)
    assert len(keywords) > 0
    assert keywords != "unknown_topic"
    
    # Test with empty texts
    empty_keywords = extract_keywords_from_cluster([])
    assert empty_keywords == "unknown_topic"
    
    # Test with single text
    single_keywords = extract_keywords_from_cluster(["Single test text"])
    assert isinstance(single_keywords, str)
    assert len(single_keywords) > 0


def test_generate_topic_labels_integration():
    """Integration test for the complete topic labeling pipeline."""
    # Create temporary events file with diverse content
    events_data = []
    
    # Add music-related events
    for i in range(10):
        events_data.append({
            "timestamp": f"2025-09-07T13:31:{52+i:02d}.916756+00:00",
            "source": "youtube",
            "type": "upload",
            "user_id": f"music_user_{i}",
            "content_id": f"music_video_{i}",
            "text": f"Artist Name - Song Title {i} [Official Music Video]"
        })
    
    # Add tech-related tweets
    for i in range(10):
        events_data.append({
            "timestamp": f"2025-09-07T13:32:{i:02d}.916756+00:00",
            "source": "twitter",
            "type": "original",
            "user_id": f"tech_user_{i}",
            "content_id": f"tech_tweet_{i}", 
            "text": f"Latest technology news about AI and machine learning {i}",
            "hashtags": ["tech", "ai", "ml"]
        })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        events_path = temp_path / "events.jsonl"
        topic_lookup_path = temp_path / "topic_lookup.json"
        
        # Write events file
        with open(events_path, 'w') as f:
            for event in events_data:
                f.write(json.dumps(event) + '\n')
        
        # Create initial topic lookup with placeholder topics
        initial_lookup = {
            "123456": "topic_0",
            "789012": "topic_1", 
            "345678": "existing_topic"  # This should not be changed
        }
        with open(topic_lookup_path, 'w') as f:
            json.dump(initial_lookup, f)
        
        # Run topic labeling
        updated_lookup = generate_topic_labels(
            events_path=events_path,
            topic_lookup_path=topic_lookup_path,
            n_clusters=3,
            max_events=20
        )
        
        # Verify results
        assert isinstance(updated_lookup, dict)
        assert len(updated_lookup) >= len(initial_lookup)
        
        # Check that placeholder topics were updated
        topic_0_updated = False
        topic_1_updated = False
        
        for topic_id, label in updated_lookup.items():
            if topic_id == "123456" and label != "topic_0":
                topic_0_updated = True
            if topic_id == "789012" and label != "topic_1": 
                topic_1_updated = True
        
        # At least some placeholder topics should be updated
        assert topic_0_updated or topic_1_updated
        
        # Existing non-placeholder topics should remain unchanged
        assert updated_lookup.get("345678") == "existing_topic"
        
        # Verify the file was actually updated
        with open(topic_lookup_path, 'r') as f:
            saved_lookup = json.load(f)
        
        assert saved_lookup == updated_lookup


if __name__ == "__main__":
    test_collect_texts_from_events()
    test_extract_keywords_from_cluster()
    test_generate_topic_labels_integration()
    print("All tests passed!")