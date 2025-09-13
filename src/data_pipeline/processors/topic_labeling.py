"""Topic labeling pipeline for generating meaningful topic names from textual examples."""

import json
import re
import hashlib
import numpy as np

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from config.config import TOPIC_LOOKUP_PATH


class TopicLabeler:
    """Generate meaningful topic labels from textual examples using clustering and text analysis."""
    
    def __init__(
        self,
        events_path: str = "data/events.jsonl",
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

    def _stable_id(name: str) -> str:
        """Deterministic 6-digit ID from a name (stable across processes)."""
        return str(int(hashlib.blake2s(name.encode("utf-8"), digest_size=4).hexdigest(), 16) % 1_000_000)
    
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
        """Collect text examples grouped by topic from events.

        If no existing mapping is present, bootstrap categories -> stable IDs so
        we can generate initial human labels and write topic_lookup.json.
        """
        topic_texts = defaultdict(list)

        # 1) Load current mapping (may be empty on first run)
        current_mapping = self.load_topic_lookup()

        # 2) Ingest texts from events (light categorization)
        text_categories = defaultdict(list)
        processed_events = 0

        if self.events_path.exists():
            with open(self.events_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    txt = (ev.get("text") or "").strip()
                    if not txt:
                        continue
                    processed_events += 1
                    low = txt.lower()
                    src = ev.get("source", "")

                    if any(k in low for k in ["music", "song"]) or any(a in txt for a in ["Swift", "Gaga", "YoungBoy", "Carpenter"]):
                        text_categories["music"].append(txt)
                    elif "video" in low and src == "youtube":
                        text_categories["video"].append(txt)
                    elif any(k in low for k in ["game", "gaming", "esports"]):
                        text_categories["gaming"].append(txt)
                    elif any(k in low for k in ["tech", "ai", "artificial", "machine learning"]):
                        text_categories["technology"].append(txt)
                    elif src == "twitter":
                        text_categories["social"].append(txt)
                    else:
                        text_categories["general"].append(txt)

        # If very few events, keep going with synthetic support text from existing labels
        if processed_events < 10 and current_mapping:
            for _, lbl in current_mapping.items():
                text_categories["synthetic"].extend([f"{lbl} trending", f"Latest {lbl} update"])

        # 3) Map categories -> topic IDs
        if current_mapping:
            # distribute categories to *existing* topic ids
            topic_ids = list(current_mapping.keys())
            cat_names = list(text_categories.keys())
            for i, cat in enumerate(cat_names):
                if i < len(topic_ids):
                    topic_texts[topic_ids[i]].extend(text_categories[cat])
        else:
            # BOOTSTRAP: create stable IDs from category names
            for cat, texts in text_categories.items():
                if not texts:
                    continue
                topic_id = self._stable_id(cat)  # e.g. "music" -> "136002"
                topic_texts[topic_id].extend(texts)

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
    events_path: str = "data/events.jsonl",
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