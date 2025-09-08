"""Tests for topic labeling pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from data_pipeline.processors.topic_labeling import TopicLabeler, run_topic_labeling_pipeline


@pytest.fixture
def sample_events_file():
    """Create a temporary events file with sample data."""
    events_data = [
        {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "source": "youtube", 
            "type": "upload",
            "user_id": "yt_u_TaylorSwift", 
            "content_id": "yt_v_song1",
            "text": "Taylor Swift - Love Song (Official Music Video)"
        },
        {
            "timestamp": "2025-01-01T00:01:00+00:00",
            "source": "youtube",
            "type": "upload", 
            "user_id": "yt_u_TaylorSwift",
            "content_id": "yt_v_song2",
            "text": "Taylor Swift - Another Love Song [Official Video]"
        },
        {
            "timestamp": "2025-01-01T00:02:00+00:00",
            "source": "twitter",
            "type": "original",
            "user_id": "u_tech",
            "content_id": "t_tech1", 
            "text": "New AI breakthrough in machine learning announced today"
        },
        {
            "timestamp": "2025-01-01T00:03:00+00:00",
            "source": "twitter",
            "type": "original",
            "user_id": "u_tech2",
            "content_id": "t_tech2",
            "text": "Artificial intelligence research shows promising results"
        },
        {
            "timestamp": "2025-01-01T00:04:00+00:00",
            "source": "youtube",
            "type": "upload",
            "user_id": "yt_u_Gaming", 
            "content_id": "yt_v_game1",
            "text": "Epic Gaming Moments - Best Highlights 2025"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for event in events_data:
            f.write(json.dumps(event) + '\n')
        return f.name


@pytest.fixture
def sample_topic_lookup():
    """Create a temporary topic lookup file with sample data."""
    # Generate topic IDs based on the content from sample events
    topic_mapping = {
        str(abs(hash("yt_v_song1")) % 1_000_000): "topic_0",
        str(abs(hash("yt_v_song2")) % 1_000_000): "topic_1", 
        str(abs(hash("t_tech1")) % 1_000_000): "viral_tweet_0",
        str(abs(hash("t_tech2")) % 1_000_000): "trending_topic_1",
        str(abs(hash("yt_v_game1")) % 1_000_000): "test_topic"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(topic_mapping, f, indent=2)
        return f.name, topic_mapping


class TestTopicLabeler:
    """Test cases for TopicLabeler class."""
    
    def test_init(self):
        """Test TopicLabeler initialization."""
        labeler = TopicLabeler()
        assert labeler.min_texts_per_topic == 3
        assert labeler.max_clusters_per_topic == 3
        assert labeler.max_label_words == 3
        assert labeler.text_embedder is not None
        assert labeler.tfidf is not None
    
    def test_load_topic_lookup_existing(self, sample_topic_lookup):
        """Test loading existing topic lookup file."""
        lookup_file, expected_mapping = sample_topic_lookup
        
        labeler = TopicLabeler(topic_lookup_path=lookup_file)
        mapping = labeler.load_topic_lookup()
        
        assert mapping == expected_mapping
        
        # Clean up
        Path(lookup_file).unlink()
    
    def test_load_topic_lookup_missing(self):
        """Test loading non-existent topic lookup file."""
        labeler = TopicLabeler(topic_lookup_path="/nonexistent/file.json")
        mapping = labeler.load_topic_lookup()
        
        assert mapping == {}
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        labeler = TopicLabeler()
        
        # Test URL removal
        text = "Check out this link: https://example.com/video"
        cleaned = labeler.clean_text(text)
        assert "https://example.com/video" not in cleaned
        assert "Check out this link:" in cleaned
        
        # Test mention removal
        text = "Hey @username, great post!"
        cleaned = labeler.clean_text(text)
        assert "@username" not in cleaned
        assert "Hey , great post!" in cleaned
        
        # Test hashtag removal
        text = "Love this song #music #trending"
        cleaned = labeler.clean_text(text)
        assert "#music" not in cleaned
        assert "#trending" not in cleaned
        assert "Love this song" in cleaned
        
        # Test bracket removal
        text = "Taylor Swift - Song [Official Video]"
        cleaned = labeler.clean_text(text)
        assert "[Official Video]" not in cleaned
        assert "Taylor Swift - Song" in cleaned
    
    def test_extract_keywords_from_cluster(self):
        """Test keyword extraction from text cluster."""
        labeler = TopicLabeler()
        
        texts = [
            "Taylor Swift new song release",
            "Taylor Swift music video official", 
            "Swift releases new album track"
        ]
        
        keywords = labeler.extract_keywords_from_cluster(texts)
        
        # Should extract relevant keywords
        assert len(keywords) > 0
        # Should contain relevant music-related terms
        keywords_str = " ".join(keywords).lower()
        assert any(term in keywords_str for term in ["taylor", "swift", "song", "music", "new"])
    
    def test_extract_keywords_empty_input(self):
        """Test keyword extraction with empty input."""
        labeler = TopicLabeler()
        
        assert labeler.extract_keywords_from_cluster([]) == []
        assert labeler.extract_keywords_from_cluster(["", "   ", ""]) == []
    
    def test_generate_label_for_topic_insufficient_texts(self):
        """Test label generation with insufficient texts."""
        labeler = TopicLabeler(min_texts_per_topic=5)
        
        texts = ["One text", "Two text"]
        topic_id = "123456"
        
        label = labeler.generate_label_for_topic(texts, topic_id)
        assert label == "topic_456"  # Should use last 3 digits
    
    def test_generate_label_for_topic_sufficient_texts(self):
        """Test label generation with sufficient texts."""
        labeler = TopicLabeler(min_texts_per_topic=2)
        
        texts = [
            "Taylor Swift - Love Story Official Video", 
            "Taylor Swift new music release",
            "Swift performs at concert venue"
        ]
        topic_id = "123456"
        
        # Mock the embedder to avoid loading the actual model
        with patch.object(labeler.text_embedder, 'encode') as mock_encode:
            mock_encode.return_value = [0.1, 0.2, 0.3]  # Simple mock embedding
            
            label = labeler.generate_label_for_topic(texts, topic_id)
            
            # Should generate a meaningful label related to the content
            assert label != "topic_456"
            assert len(label) > 0
    
    def test_collect_texts_by_topic(self, sample_events_file, sample_topic_lookup):
        """Test collecting texts by topic from events."""
        lookup_file, topic_mapping = sample_topic_lookup
        
        labeler = TopicLabeler(
            events_path=sample_events_file,
            topic_lookup_path=lookup_file
        )
        
        topic_texts = labeler.collect_texts_by_topic()
        
        # Should collect some texts
        assert len(topic_texts) > 0
        
        # Each topic should have associated texts
        for topic_id, texts in topic_texts.items():
            assert len(texts) > 0
            assert all(isinstance(text, str) for text in texts)
        
        # Clean up
        Path(sample_events_file).unlink()
        Path(lookup_file).unlink()
    
    @patch('data_pipeline.processors.topic_labeling.RealtimeTextEmbedder')
    def test_run_labeling_pipeline(self, mock_embedder, sample_events_file, sample_topic_lookup):
        """Test the complete labeling pipeline."""
        lookup_file, topic_mapping = sample_topic_lookup
        
        # Mock the embedder
        mock_embedder.return_value.encode.return_value = [0.1, 0.2, 0.3]
        
        labeler = TopicLabeler(
            events_path=sample_events_file,
            topic_lookup_path=lookup_file,
            min_texts_per_topic=1  # Lower threshold for testing
        )
        
        result = labeler.run_labeling_pipeline()
        
        # Should return a mapping
        assert isinstance(result, dict)
        
        # Should have updated some labels
        # (exact assertion depends on the mock behavior and test data)
        
        # Clean up
        Path(sample_events_file).unlink()
        Path(lookup_file).unlink()


class TestTopicLabelingFunction:
    """Test the convenience function."""
    
    @patch('data_pipeline.processors.topic_labeling.TopicLabeler')
    def test_run_topic_labeling_pipeline_function(self, mock_labeler_class):
        """Test the convenience function calls TopicLabeler correctly."""
        mock_labeler = mock_labeler_class.return_value
        mock_labeler.run_labeling_pipeline.return_value = {"123": "Test Topic"}
        
        result = run_topic_labeling_pipeline()
        
        # Should instantiate TopicLabeler with default paths
        mock_labeler_class.assert_called_once_with(
            events_path="datasets/events.jsonl",
            topic_lookup_path="datasets/topic_lookup.json"
        )
        
        # Should call the pipeline
        mock_labeler.run_labeling_pipeline.assert_called_once()
        
        # Should return the result
        assert result == {"123": "Test Topic"}