"""Topic labeling pipeline for generating meaningful topic names from textual examples."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from config.config import TOPIC_LOOKUP_PATH


class TopicLabeler:
    """Generate meaningful topic labels from textual examples using clustering and text analysis."""
    
    def __init__(
        self,
        events_path: str = "datasets/events.jsonl",
        topic_lookup_path: str = TOPIC_LOOKUP_PATH,
        min_texts_per_topic: int = 3,
        max_clusters_per_topic: int = 3,
        max_label_words: int = 3,
        use_embedder: bool = False
    ):
        """Initialize the topic labeler.
        
        Args:
            events_path: Path to events JSONL file
            topic_lookup_path: Path to topic lookup JSON file
            min_texts_per_topic: Minimum texts needed to generate meaningful labels
            max_clusters_per_topic: Maximum subclusters per topic for label generation
            max_label_words: Maximum words in generated labels
            use_embedder: Whether to use text embedder (requires internet connection)
        """
        self.events_path = Path(events_path)
        self.topic_lookup_path = Path(topic_lookup_path)
        self.min_texts_per_topic = min_texts_per_topic
        self.max_clusters_per_topic = max_clusters_per_topic
        self.max_label_words = max_label_words
        self.use_embedder = use_embedder
        
        # Initialize text processor only if requested
        self.text_embedder = None
        if use_embedder:
            try:
                self.text_embedder = RealtimeTextEmbedder(device="cpu")
            except Exception as e:
                print(f"Warning: Could not initialize text embedder: {e}")
                print("Falling back to TF-IDF only approach")
                self.use_embedder = False
        
        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
    
    def load_topic_lookup(self) -> Dict[str, str]:
        """Load current topic lookup mapping."""
        try:
            with open(self.topic_lookup_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_topic_lookup(self, topic_mapping: Dict[str, str]) -> None:
        """Save updated topic lookup mapping."""
        with open(self.topic_lookup_path, 'w', encoding='utf-8') as f:
            json.dump(topic_mapping, f, indent=2)
    
    def collect_texts_by_topic(self) -> Dict[str, List[str]]:
        """Collect text examples grouped by topic from events."""
        topic_texts = defaultdict(list)
        
        if not self.events_path.exists():
            print(f"Events file {self.events_path} not found")
            return topic_texts
        
        # Load current topic mapping to get topic IDs
        current_mapping = self.load_topic_lookup()
        
        # Strategy: Group texts by semantic similarity and map to existing placeholder topics
        # This simulates how topics might be discovered in a real system
        
        # First, collect all texts and categorize them
        all_texts = []
        text_categories = defaultdict(list)
        
        with open(self.events_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    text = event.get('text', '').strip()
                    source = event.get('source', '')
                    event_type = event.get('type', '')
                    
                    if not text:
                        continue
                    
                    all_texts.append(text)
                    
                    # Categorize texts by source and content patterns
                    if 'music' in text.lower() or 'song' in text.lower() or any(artist in text for artist in ['Swift', 'Gaga', 'YoungBoy']):
                        text_categories['music'].append(text)
                    elif 'video' in text.lower() and source == 'youtube':
                        text_categories['video'].append(text) 
                    elif source == 'twitter' and ('tweet' in text.lower() or 'simulated' in text.lower()):
                        text_categories['social'].append(text)
                    elif 'game' in text.lower() or 'gaming' in text.lower():
                        text_categories['gaming'].append(text)
                    elif 'tech' in text.lower() or 'ai' in text.lower() or 'artificial' in text.lower():
                        text_categories['technology'].append(text)
                    else:
                        text_categories['general'].append(text)
                        
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Map categories to existing topic IDs
        # Use a deterministic assignment based on topic placeholders
        topic_assignments = {}
        topic_ids = list(current_mapping.keys())
        
        category_names = list(text_categories.keys())
        for i, category in enumerate(category_names):
            if i < len(topic_ids):
                # Assign categories to topics deterministically
                topic_id = topic_ids[i % len(topic_ids)]
                topic_assignments[category] = topic_id
        
        # Distribute texts to topics
        for category, texts in text_categories.items():
            if category in topic_assignments:
                topic_id = topic_assignments[category]
                topic_texts[topic_id].extend(texts)
            
        # Add some texts to multiple topics to simulate overlap
        if len(topic_ids) > len(category_names):
            remaining_topics = topic_ids[len(category_names):]
            all_category_texts = []
            for texts in text_categories.values():
                all_category_texts.extend(texts)
            
            # Distribute remaining texts
            for i, topic_id in enumerate(remaining_topics):
                start_idx = (i * len(all_category_texts)) // len(remaining_topics)
                end_idx = ((i + 1) * len(all_category_texts)) // len(remaining_topics)
                topic_texts[topic_id].extend(all_category_texts[start_idx:end_idx])
        
        return topic_texts
    
    def clean_text(self, text: str) -> str:
        """Clean text for better processing."""
        # Remove URLs, mentions, hashtags for cleaner processing
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Official Video] etc
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()
    
    def extract_keywords_from_cluster(self, texts: List[str]) -> List[str]:
        """Extract meaningful keywords from a cluster of texts."""
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        cleaned_texts = [t for t in cleaned_texts if t]  # Remove empty strings
        
        if not cleaned_texts:
            return []
        
        try:
            # Use TF-IDF to find important terms
            tfidf_matrix = self.tfidf.fit_transform(cleaned_texts)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get mean TF-IDF scores for each term
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top terms
            top_indices = np.argsort(mean_scores)[-10:][::-1]  # Top 10 terms
            keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0.1]
            
            return keywords[:5]  # Return top 5 keywords
            
        except Exception as e:
            # Fallback: extract common words manually
            all_words = []
            for text in cleaned_texts:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                all_words.extend(words)
            
            # Count word frequencies
            word_counts = defaultdict(int)
            for word in all_words:
                if len(word) >= 3:  # Only words with 3+ characters
                    word_counts[word] += 1
            
            # Return most frequent words
            return [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def generate_label_for_topic(self, texts: List[str], topic_id: str) -> str:
        """Generate a meaningful label for a topic based on its texts."""
        if len(texts) < self.min_texts_per_topic:
            return f"topic_{topic_id[-3:]}"  # Fallback to generic name
        
        # Clean and filter texts
        cleaned_texts = [self.clean_text(text) for text in texts[:50]]  # Limit for performance
        cleaned_texts = [t for t in cleaned_texts if t and len(t) > 10]
        
        if not cleaned_texts:
            return f"topic_{topic_id[-3:]}"
        
        try:
            # Extract keywords from all texts
            keywords = self.extract_keywords_from_cluster(cleaned_texts)
            
            if keywords:
                # Clean up keywords and create label
                label_words = []
                seen_words = set()
                
                for keyword in keywords:
                    # Clean keyword
                    clean_keyword = re.sub(r'[^a-zA-Z0-9\s]', '', keyword).strip()
                    if not clean_keyword or len(clean_keyword) < 3:
                        continue
                        
                    # Split compound keywords and take the most meaningful part
                    word_parts = clean_keyword.lower().split()
                    for part in word_parts:
                        if (len(part) >= 3 and 
                            part not in seen_words and 
                            part not in ['the', 'and', 'for', 'with', 'video', 'official', 'new'] and
                            len(label_words) < self.max_label_words):
                            label_words.append(part.capitalize())
                            seen_words.add(part)
                            break
                
                if label_words:
                    return " ".join(label_words)
            
            # Fallback: try to extract meaningful patterns from raw texts
            artist_names = set()
            song_titles = set()
            brand_names = set()
            
            for text in cleaned_texts[:10]:
                # Look for music patterns: "Artist - Song"
                music_match = re.search(r'^([^-\[\(]+)\s*[-\[\(]', text)
                if music_match:
                    artist = music_match.group(1).strip()
                    if len(artist) <= 20 and not any(skip in artist.lower() for skip in ['official', 'video', 'ft']):
                        artist_names.add(artist.title())
                
                # Look for YouTube/content patterns
                if 'Official' in text and ('Video' in text or 'Music' in text):
                    # Extract the main subject before "Official"
                    main_part = text.split('Official')[0].strip()
                    if main_part and len(main_part) <= 30:
                        main_words = main_part.split()[:2]  # Take first 2 words
                        if main_words:
                            brand_names.add(" ".join(main_words).title())
            
            # Choose the best label source
            if artist_names and len(artist_names) <= 3:
                return list(artist_names)[0][:20]  # Take first artist, limit length
            elif brand_names and len(brand_names) <= 3:
                return list(brand_names)[0][:20]  # Take first brand, limit length
            elif song_titles and len(song_titles) <= 3:
                return list(song_titles)[0][:20]  # Take first song, limit length
            
            # Final fallback: generate descriptive label based on content type
            text_lower = " ".join(cleaned_texts[:5]).lower()
            if 'music' in text_lower or 'song' in text_lower:
                return "Music"
            elif 'game' in text_lower or 'gaming' in text_lower:
                return "Gaming"  
            elif 'tweet' in text_lower or 'twitter' in text_lower:
                return "Social Media"
            elif 'video' in text_lower and 'youtube' in text_lower:
                return "Video Content"
            elif 'tech' in text_lower or 'ai' in text_lower:
                return "Technology"
            else:
                return "General Content"
                
        except Exception as e:
            print(f"Error generating label for topic {topic_id}: {e}")
        
        # Ultimate fallback
        return f"topic_{topic_id[-3:]}"
    
    def run_labeling_pipeline(self) -> Dict[str, str]:
        """Run the complete topic labeling pipeline."""
        print("Starting topic labeling pipeline...")
        
        # Load current topic mapping
        current_mapping = self.load_topic_lookup()
        print(f"Found {len(current_mapping)} existing topics")
        
        # Collect texts for each topic
        print("Collecting texts by topic...")
        topic_texts = self.collect_texts_by_topic()
        print(f"Collected texts for {len(topic_texts)} topics")
        
        # Generate new labels
        updated_mapping = current_mapping.copy()
        updated_count = 0
        
        for topic_id, texts in topic_texts.items():
            if len(texts) >= self.min_texts_per_topic:
                current_label = current_mapping.get(topic_id, "")
                
                # Only update if current label is placeholder-like
                if (not current_label or 
                    current_label.startswith("topic_") or 
                    current_label.startswith("test_") or
                    current_label.startswith("viral_") or
                    current_label.startswith("trending_")):
                    
                    new_label = self.generate_label_for_topic(texts, topic_id)
                    if new_label != current_label:
                        updated_mapping[topic_id] = new_label
                        updated_count += 1
                        print(f"Updated topic {topic_id}: '{current_label}' -> '{new_label}'")
        
        print(f"Updated {updated_count} topic labels")
        
        # Save updated mapping
        if updated_count > 0:
            self.save_topic_lookup(updated_mapping)
            print(f"Saved updated topic mapping to {self.topic_lookup_path}")
        
        return updated_mapping


def run_topic_labeling_pipeline(
    events_path: str = "datasets/events.jsonl",
    topic_lookup_path: str = TOPIC_LOOKUP_PATH,
    use_embedder: bool = False
) -> Dict[str, str]:
    """Convenience function to run the topic labeling pipeline."""
    labeler = TopicLabeler(
        events_path=events_path, 
        topic_lookup_path=topic_lookup_path,
        use_embedder=use_embedder
    )
    return labeler.run_labeling_pipeline()


if __name__ == "__main__":
    # Run the pipeline when executed directly
    result = run_topic_labeling_pipeline()
    print(f"Pipeline completed. Updated {len(result)} topics.")