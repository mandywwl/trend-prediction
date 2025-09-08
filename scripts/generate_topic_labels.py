#!/usr/bin/env python3
"""
Generate meaningful topic labels from event data and update topic_lookup.json.

This script:
1. Collects textual examples from events.jsonl
2. Embeds texts using RealtimeTextEmbedder  
3. Clusters similar texts using K-means
4. Generates labels using TF-IDF keyword extraction
5. Updates topic_lookup.json with the new labels
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils.path_utils import find_repo_root
except ImportError:
    def find_repo_root():
        return Path(__file__).parent.parent


def collect_texts_from_events(events_path: Path, max_events: int = None) -> List[str]:
    """Collect text content from events for clustering."""
    texts = []
    
    with open(events_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_events and i >= max_events:
                break
                
            event = json.loads(line.strip())
            text_content = []
            
            # Extract main text
            if 'text' in event and event['text']:
                text_content.append(event['text'])
            
            # Extract hashtags if available
            if 'hashtags' in event and event['hashtags']:
                hashtag_text = ' '.join(f"#{tag}" for tag in event['hashtags'])
                text_content.append(hashtag_text)
            
            # Combine text elements
            if text_content:
                full_text = ' '.join(text_content)
                # Clean text - remove extra whitespace and normalize
                clean_text = re.sub(r'\s+', ' ', full_text).strip()
                if clean_text and len(clean_text) > 5:  # Filter out very short texts
                    texts.append(clean_text)
    
    return texts


def extract_keywords_from_cluster(texts: List[str], top_k: int = 3) -> str:
    """Extract representative keywords from a cluster of texts using TF-IDF."""
    if not texts:
        return "unknown_topic"
    
    # Combine all texts in cluster for keyword extraction
    combined_text = ' '.join(texts)
    
    # Use TF-IDF to find important words
    vectorizer = TfidfVectorizer(
        max_features=50,
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=1,
        lowercase=True
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Get top keywords by TF-IDF score
        top_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
        keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
        
        if keywords:
            # Create label from top keywords
            label = '_'.join(keywords[:2])  # Use top 2 keywords
            # Clean label - remove special characters, limit length
            label = re.sub(r'[^a-zA-Z0-9_]', '', label)[:20]
            return label.lower() if label else "cluster_topic"
        
    except Exception as e:
        print(f"Error extracting keywords: {e}")
    
    # Fallback: use most common words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
    if words:
        word_counts = Counter(words)
        common_words = [word for word, _ in word_counts.most_common(2)]
        return '_'.join(common_words[:2])
    
    return "unlabeled_topic"


def generate_topic_labels(
    events_path: Path, 
    topic_lookup_path: Path,
    n_clusters: int = 10,
    max_events: int = 500
) -> Dict[str, str]:
    """Generate meaningful topic labels and return updated mappings."""
    
    print("Collecting texts from events...")
    texts = collect_texts_from_events(events_path, max_events)
    print(f"Collected {len(texts)} texts for clustering")
    
    if len(texts) < n_clusters:
        print(f"Warning: Only {len(texts)} texts available, reducing clusters to {len(texts)}")
        n_clusters = min(n_clusters, len(texts))
    
    if len(texts) == 0:
        print("No texts found in events")
        return {}
    
    # Use TF-IDF vectorization instead of deep embeddings
    print("Vectorizing texts with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        lowercase=True
    )
    
    # Create TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"Generated TF-IDF matrix with shape: {tfidf_matrix.shape}")
    
    # Cluster using TF-IDF vectors
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
    
    # Group texts by cluster
    clusters = defaultdict(list)
    for text, label in zip(texts, cluster_labels):
        clusters[label].append(text)
    
    # Generate labels for each cluster
    print("Generating cluster labels...")
    cluster_labels_map = {}
    for cluster_id, cluster_texts in clusters.items():
        label = extract_keywords_from_cluster(cluster_texts)
        cluster_labels_map[cluster_id] = label
        print(f"Cluster {cluster_id} ({len(cluster_texts)} texts) -> '{label}'")
        # Show sample texts
        sample_texts = cluster_texts[:2]
        for i, text in enumerate(sample_texts):
            print(f"  Sample {i+1}: {text[:60]}...")
    
    # Load existing topic lookup 
    print(f"Loading existing topic lookup from {topic_lookup_path}")
    if topic_lookup_path.exists():
        with open(topic_lookup_path, 'r', encoding='utf-8') as f:
            existing_lookup = json.load(f)
    else:
        existing_lookup = {}
    
    # Create new mappings
    updated_lookup = existing_lookup.copy()
    
    # Update placeholder topics with generated labels
    placeholder_count = 0
    used_clusters = set()
    
    for topic_id, current_label in existing_lookup.items():
        # Check if this is a placeholder topic
        if current_label.startswith('topic_') and current_label.count('_') == 1:
            try:
                # Extract topic number and map to cluster
                topic_num = int(current_label.split('_')[1])
                if topic_num < len(cluster_labels_map):
                    new_label = cluster_labels_map[topic_num]
                    updated_lookup[topic_id] = new_label
                    placeholder_count += 1
                    used_clusters.add(topic_num)
                    print(f"Updated {current_label} -> {new_label}")
            except (ValueError, IndexError):
                continue
    
    # Add any remaining cluster labels as additional topics (only if not used)
    for cluster_id, label in cluster_labels_map.items():
        if cluster_id not in used_clusters:
            # Create a new topic ID for this cluster
            new_topic_id = str(abs(hash(f"cluster_{cluster_id}")) % 1_000_000)
            # Make sure we don't override existing topics
            while new_topic_id in updated_lookup:
                new_topic_id = str((int(new_topic_id) + 1) % 1_000_000)
            updated_lookup[new_topic_id] = label
    
    print(f"Updated {placeholder_count} placeholder topics")
    print(f"Total topics in lookup: {len(updated_lookup)}")
    
    # Save the updated lookup to file
    with open(topic_lookup_path, 'w', encoding='utf-8') as f:
        json.dump(updated_lookup, f, indent=2)
    
    return updated_lookup


def main():
    """Main function to run the topic labeling pipeline."""
    
    # Get repository paths
    try:
        repo_root = find_repo_root()
    except:
        repo_root = Path(__file__).parent.parent
    
    events_path = repo_root / "datasets" / "events.jsonl"
    topic_lookup_path = repo_root / "datasets" / "topic_lookup.json"
    
    if not events_path.exists():
        print(f"Error: Events file not found at {events_path}")
        return 1
    
    print(f"Starting topic labeling pipeline...")
    print(f"Events file: {events_path}")
    print(f"Topic lookup file: {topic_lookup_path}")
    
    # Generate updated topic labels
    try:
        updated_lookup = generate_topic_labels(
            events_path=events_path,
            topic_lookup_path=topic_lookup_path,
            n_clusters=8,  # Reasonable number for initial clustering
            max_events=400  # Process subset for efficiency
        )
        
        if updated_lookup:
            # Save updated topic lookup
            print(f"Saving updated topic lookup to {topic_lookup_path}")
            with open(topic_lookup_path, 'w', encoding='utf-8') as f:
                json.dump(updated_lookup, f, indent=2)
            
            print("Topic labeling pipeline completed successfully!")
            
            # Show some results
            print("\nSample updated topics:")
            for i, (topic_id, label) in enumerate(list(updated_lookup.items())[:5]):
                print(f"  {topic_id}: {label}")
            
        else:
            print("No topics generated")
            return 1
            
    except Exception as e:
        print(f"Error in topic labeling pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())