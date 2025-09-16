"""Topic labeling pipeline for generating meaningful topic names from textual examples."""

import json
import re
import hashlib
import numpy as np

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from config.config import LABELING_JUNK_FILTER, TOPIC_LOOKUP_PATH


class TopicLabeler:
    """Generate meaningful topic labels from textual examples using clustering and text analysis."""
    
    def __init__(
        self,
        events_path: str = "data/events.jsonl",
        topic_lookup_path: str = TOPIC_LOOKUP_PATH,
        min_texts_per_topic: int = 2,
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
            min_df=1  # relax for small corpora
        )

        # Build a unified, normalized junk set once
        cfg_junk = set(LABELING_JUNK_FILTER or [])
        # Optional: local additions you want to enforce
        local_junk = {
            "viral","fyp","short","shorts","subscribe","channel","watch","like","share",
            "clip","clips","live","stream","follow",
            "official","video","new","latest","trending","best","top",
            "just","now","today",
            "overrated","fight","hot","hottake",
            # broad buckets you don't want as titles by themselves
            "music","gaming","technology","social"
        }
        # normalize to lowercase and strip
        self.junk = {w.strip().lower() for w in (cfg_junk | local_junk)}

    @staticmethod
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

        We group by a **stable topic key**:
          1) prefer explicit `topic_id` if present in event,
          2) else a canonical hashtag/entity field,
          3) else a deterministic hash of a fallback key (content_id/text).
        """
        topic_texts: Dict[str, List[str]] = defaultdict(list)
        if not self.events_path.exists():
            return topic_texts
        
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                
                raw_txt = (ev.get("text") or "").strip()
                if not raw_txt:
                    continue
                txt = self.clean_text(raw_txt)
                if not txt:
                    continue

                # stable topic id resolution
                topic_id = None
                if "topic_id" in ev and ev["topic_id"] is not None:
                    topic_id = str(ev["topic_id"])
                else:
                    # Try canonical hashtag/entity keys  kept in events
                    topic_key = (
                        ev.get("hashtag")
                        or ev.get("entity")
                        or ev.get("topic")
                        or ev.get("content_id")
                        or raw_txt[:64]  # fallback to text if nothing else
                    )
                    topic_id = self._stable_id(str(topic_key))

                topic_texts[topic_id].append(txt)
        return topic_texts

    @staticmethod
    def clean_text(s: str) -> str:
        """Clean text for better processing."""
        s = re.sub(r'http\S+', ' ', s)
        s = re.sub(r'@\w+', ' ', s)
        # Convert hashtags instead of dropping them
        s = re.sub(r'#([A-Za-z0-9_]+)', r'\1', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()
    
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
        
    def _vectorize_and_cluster(self, topic_texts: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Build per-topic documents, TF-IDF vectorise, cluster with KMeans,
        then name clusters from top TF-IDF terms. Returns {topic_id -> label}.
        """
        # build per-topic docs and filter by minimum support
        docs = []
        tids = []
        for tid, texts in topic_texts.items():
            good = [self.clean_text(t) for t in texts if isinstance(t, str) and t.strip()]
            if len(good) < self.min_texts_per_topic:
                continue
            doc = " ".join(good)
            if not doc:
                continue
            tids.append(tid)
            docs.append(doc)
        if not docs:
            return {}
        
        # TF-IDF over per-topic documents (not per-message)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1, # drop extremely rare terms
            max_df=0.8, # drop very common terms
            max_features=20000,
            sublinear_tf=True, # dampen large counts
            smooth_idf=True  # avoid zero idf
        )
        X = vectorizer.fit_transform(docs)
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        # choose K by a simple heuristic (more clusters → more specific labels)
        # ensure 2 ≤ K ≤ n_docs, target ~1 cluster per 20 topics, min 5 when enough data
        if len(tids) < 10:
            K = min(2, len(tids))
        else:
            K = max(5, min(len(tids), len(tids) // 20))
        kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        
        # build cluster label strings from centroid/top terms
        cluster_terms: Dict[int, List[str]] = {}
        junk = self.junk

        def _is_bad(term: str) -> bool:
            if not term or len(term) < 3 or term.isnumeric():
                return True
            # check each token inside the n-gram
            parts = re.split(r"[\s_]+", term.lower())
            return any(p in junk for p in parts)

        for c in range(K):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            centroid = X[idx].mean(axis=0).A1
            top_idx = centroid.argsort()[-20:][::-1]  # more candidates, we'll filter down
            candidates = [feature_names[i] for i in top_idx if not _is_bad(feature_names[i])]
            # prefer bigrams first, then fill with unigrams
            bigrams = [t for t in candidates if " " in t]
            unigrams = [t for t in candidates if " " not in t]
            terms = (bigrams[:3] + unigrams[:2]) or unigrams[:3]
            cluster_terms[c] = terms[:5]

        def mk_label(terms: List[str]) -> str:
            if not terms:
                return "Topic"
            # prefer 2–3 best terms, title-cased
            chosen = terms[:3]
            return " ".join([t.title() for t in chosen])
        
        cluster_label = {c: mk_label(cluster_terms.get(c, [])) for c in range(K)}
        # map each topic_id to its cluster label, and avoid bare "Topic"
        topic_label_map: Dict[str, str] = {}
        for i, tid in enumerate(tids):
            lbl = cluster_label.get(labels[i], "Topic")
            if lbl == "Topic":
                # last-resort unique label so it won't collapse with others
                lbl = f"Topic {str(tid)[-6:]}"
            topic_label_map[tid] = lbl
        return topic_label_map
    
    
    def run_labeling_pipeline(self) -> Dict[str, str]:
        """Run the complete topic labeling pipeline."""
        print("Starting topic labeling pipeline...")

        try: 
            current_mapping = self.load_topic_lookup()
            print(f"Found {len(current_mapping)} existing topics")

            print("Collecting texts by topic...")
            topic_texts = self.collect_texts_by_topic()
            print(f"Collected texts for {len(topic_texts)} topics")

            # Build labels via TF-IDF + KMeans clustering
            print("Vectorising and clustering per-topic documents...")
            new_labels = self._vectorize_and_cluster(topic_texts)
            print(f"Generated labels for {len(new_labels)} topics via clustering")

            # Merge: keep existing non-placeholder labels; fill placeholders/new topics from clustering
            def is_placeholder(lbl: str) -> bool:
                return (not lbl) or lbl.startswith(("topic_", "test_", "viral_", "trending_")) or str(lbl).isdigit()
            
            updated = current_mapping.copy()
            updates = 0
            for tid, lbl in new_labels.items():
                if tid not in updated or is_placeholder(str(updated.get(tid, ""))):
                    if tid not in updated or is_placeholder(str(updated.get(tid, ""))):
                        if lbl and lbl != updated.get(tid, ""):
                            updated[tid] = lbl
                            updates += 1
            print(f"Updated {updates} topic labels")
            if updates > 0:
                self.save_topic_lookup(updated)
                print(f"Saved updated topic mapping to {self.topic_lookup_path}")
            return updated
        except Exception as e:
            print(f"[topic_labeling] Error running topic labeling pipeline: {e}")
            return {}



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