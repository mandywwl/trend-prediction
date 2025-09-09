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
            print(f"Events file {self.events_path} not found, creating from existing topic labels")
            # Generate synthetic text examples based on existing topic labels
            current_mapping = self.load_topic_lookup()
            return self._generate_synthetic_texts(current_mapping)
        
        # Load current topic mapping to get topic IDs
        current_mapping = self.load_topic_lookup()
        
        # Strategy: Group texts by semantic similarity and map to existing placeholder topics
        # This simulates how topics might be discovered in a real system
        
        # First, collect all texts and categorize them
        all_texts = []
        text_categories = defaultdict(list)
        
        processed_events = 0
        with open(self.events_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    text = event.get('text', '').strip()
                    source = event.get('source', '')
                    event_type = event.get('type', '')
                    
                    if not text:
                        continue
                    
                    processed_events += 1
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
        
        # If we processed very few events, supplement with synthetic data
        if processed_events < 10:
            print(f"Only {processed_events} events found, supplementing with synthetic data")
            synthetic_texts = self._generate_synthetic_texts(current_mapping)
            for topic_id, texts in synthetic_texts.items():
                text_categories['synthetic'].extend(texts)
        
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
    
    def _generate_synthetic_texts(self, current_mapping: Dict[str, str]) -> Dict[str, List[str]]:
        """Generate synthetic text examples based on existing topic labels."""
        topic_texts = defaultdict(list)
        
        for topic_id, label in current_mapping.items():
            if not label or label.startswith(('topic_', 'test_', 'viral_', 'trending_')):
                continue
                
            # Generate synthetic texts based on the label
            synthetic_texts = []
            if 'music' in label.lower() or any(name in label for name in ['Weeknd', 'Swift', 'YoungBoy', 'Carpenter']):
                synthetic_texts = [
                    f"{label} - Official Music Video",
                    f"New song by {label} trending now",
                    f"{label} latest album release",
                ]
            elif 'tech' in label.lower() or 'ai' in label.lower():
                synthetic_texts = [
                    f"New {label} technology breakthrough",
                    f"{label} artificial intelligence update",
                    f"Latest {label} tech innovation",
                ]
            elif 'game' in label.lower():
                synthetic_texts = [
                    f"{label} gaming tournament results",
                    f"New {label} game release",
                    f"{label} esports championship",
                ]
            else:
                # Generic content
                synthetic_texts = [
                    f"{label} trending content",
                    f"Latest {label} update",
                    f"{label} viral post",
                ]
            
            topic_texts[topic_id] = synthetic_texts
        
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
            # First pass: Look for specific recognizable patterns
            text_sample = " ".join(cleaned_texts[:10]).lower()
            
            # Extract artist/performer names from music content
            artist_names = set()
            song_titles = set()
            content_types = []
            
            for text in cleaned_texts[:15]:
                original_text = text
                text_lower = text.lower()
                
                # Music video patterns
                if any(pattern in text_lower for pattern in ['official video', 'music video', 'vevo']):
                    # Extract artist name before " - " or " |"
                    for sep in [' - ', ' | ', ' (Official']:
                        if sep in original_text:
                            potential_artist = original_text.split(sep)[0].strip()
                            if (len(potential_artist) <= 30 and 
                                not any(skip in potential_artist.lower() for skip in 
                                       ['youtube', 'official', 'video', 'music', 'ft.', 'feat'])):
                                artist_names.add(potential_artist)
                                break
                
                # Sports content
                if any(term in text_lower for term in ['nfl', 'nba', 'soccer', 'football', 'baseball', 'basketball']):
                    content_types.append('Sports')
                
                # Gaming content  
                if any(term in text_lower for term in ['game', 'gaming', 'esports', 'tournament', 'gameplay']):
                    content_types.append('Gaming')
                    
                # Tech content
                if any(term in text_lower for term in ['ai', 'tech', 'software', 'app', 'device', 'iphone', 'android']):
                    content_types.append('Technology')
                    
                # News/Politics
                if any(term in text_lower for term in ['trump', 'biden', 'election', 'politics', 'news']):
                    content_types.append('News')
                    
                # Social media/viral content
                if any(term in text_lower for term in ['viral', 'trending', 'tiktok', 'fyp', 'meme']):
                    content_types.append('Viral Content')
                    
                # Entertainment/Celebrity
                if any(term in text_lower for term in ['celebrity', 'hollywood', 'actor', 'actress', 'movie', 'film']):
                    content_types.append('Entertainment')
            
            # Choose best label based on extracted information
            
            # 1. If we found specific artists, use the most common one
            if artist_names:
                # Take the most frequent or first artist name
                best_artist = list(artist_names)[0]
                if len(best_artist) <= 25:  # Reasonable length
                    return best_artist
            
            # 2. If we have a clear content type, use it
            if content_types:
                # Get most common content type
                from collections import Counter
                content_counter = Counter(content_types)
                most_common_type = content_counter.most_common(1)[0][0]
                return most_common_type
            
            # 3. Try TF-IDF keyword extraction
            keywords = self.extract_keywords_from_cluster(cleaned_texts)
            if keywords:
                # Filter out common words and create meaningful label
                meaningful_keywords = []
                for keyword in keywords[:3]:
                    clean_keyword = re.sub(r'[^a-zA-Z0-9\s]', '', keyword).strip()
                    if (clean_keyword and 
                        len(clean_keyword) >= 3 and
                        clean_keyword.lower() not in ['video', 'official', 'music', 'new', 'the', 'and', 'for']):
                        meaningful_keywords.append(clean_keyword.capitalize())
                
                if meaningful_keywords:
                    return " ".join(meaningful_keywords[:2])  # Take top 2 meaningful keywords
            
            # 4. Pattern-based extraction for specific content
            for text in cleaned_texts[:5]:
                # Look for branded content patterns  
                if 'vs' in text.lower() or 'versus' in text.lower():
                    return "Competition"
                if any(brand in text.lower() for brand in ['apple', 'google', 'microsoft', 'tesla', 'netflix']):
                    return "Tech Brands"
                if re.search(r'\b(recipe|cooking|food|chef)\b', text.lower()):
                    return "Food & Cooking"
                if re.search(r'\b(workout|fitness|gym|health)\b', text.lower()):
                    return "Fitness"
            
            # 5. Fallback to content type based on overall theme
            combined_text = " ".join(cleaned_texts[:5]).lower()
            if 'simulated' in combined_text and 'tweet' in combined_text:
                return "Social Posts"
            elif any(word in combined_text for word in ['music', 'song', 'album', 'artist']):
                return "Music"
            elif any(word in combined_text for word in ['video', 'youtube', 'watch']):
                return "Video Content"
            elif any(word in combined_text for word in ['game', 'play', 'gaming']):
                return "Gaming"
            elif any(word in combined_text for word in ['news', 'breaking', 'report']):
                return "News"
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
        
        # Clean up circular references where topic IDs point to other topic IDs
        print("Cleaning up circular references...")
        cleaned_mapping = current_mapping.copy()
        cleanup_count = 0
        
        for topic_id, label in current_mapping.items():
            # If label is another topic ID, try to resolve it
            if label in current_mapping and label != topic_id:
                resolved_label = current_mapping[label]
                # Only resolve if the target has a meaningful label
                if (resolved_label and 
                    not resolved_label.isdigit() and 
                    not resolved_label.startswith("topic_") and
                    resolved_label not in current_mapping):  # Avoid double references
                    cleaned_mapping[topic_id] = resolved_label
                    cleanup_count += 1
                    print(f"Resolved circular reference {topic_id}: '{label}' -> '{resolved_label}'")
        
        print(f"Cleaned up {cleanup_count} circular references")
        
        # Collect texts for each topic
        print("Collecting texts by topic...")
        topic_texts = self.collect_texts_by_topic()
        print(f"Collected texts for {len(topic_texts)} topics")
        
        # Generate new labels
        updated_mapping = cleaned_mapping.copy()
        updated_count = 0
        
        for topic_id, texts in topic_texts.items():
            if len(texts) >= self.min_texts_per_topic:
                current_label = cleaned_mapping.get(topic_id, "")
                
                # Check if current label is meaningless and should be updated
                # This includes: empty labels, placeholder labels, pure numeric strings, or topic IDs as labels
                is_meaningless_label = (
                    not current_label or 
                    current_label.startswith("topic_") or 
                    current_label.startswith("test_") or
                    current_label.startswith("viral_") or
                    current_label.startswith("trending_") or
                    current_label.isdigit() or  # Pure numeric labels (hashed IDs)
                    (len(current_label) >= 5 and current_label.isdigit()) or  # Long numeric strings
                    current_label in cleaned_mapping  # Still pointing to another topic ID
                )
                
                if is_meaningless_label:
                    new_label = self.generate_label_for_topic(texts, topic_id)
                    if new_label != current_label and not new_label.startswith("topic_"):
                        updated_mapping[topic_id] = new_label
                        updated_count += 1
                        print(f"Updated topic {topic_id}: '{current_label}' -> '{new_label}'")
        
        print(f"Updated {updated_count} topic labels")
        
        # Save updated mapping
        if updated_count > 0 or cleanup_count > 0:
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