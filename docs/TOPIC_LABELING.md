# Topic Labeling Pipeline

## Overview

The Topic Labeling Pipeline automatically generates meaningful topic names by analyzing textual examples from social media events, replacing placeholder strings in `datasets/topic_lookup.json` with semantic labels derived from content analysis.

## Features

### 1. Textual Example Gathering
- Collects text examples from `datasets/events.jsonl`
- Categorizes content by source and semantic patterns
- Groups texts into meaningful categories (music, technology, gaming, social media, etc.)

### 2. Semantic Clustering  
- Uses TF-IDF vectorization for keyword extraction
- Optionally supports DistilBERT embeddings for advanced clustering
- Identifies representative text clusters within each topic

### 3. Label Derivation
- Extracts meaningful keywords using multiple strategies:
  - Artist/brand name detection (e.g., "Taylor Swift", "YoungBoy Never Broke Again")
  - Content type classification (e.g., "Technology", "Gaming", "Music")
  - TF-IDF keyword extraction for unique terms
- Generates clean, concise labels (max 3 words)

### 4. Automatic Integration
- Updates `datasets/topic_lookup.json` with new labels
- Integrates seamlessly with existing dashboard display
- Preserves topic ID mappings for consistency

## Usage

### Standalone Execution

```bash
# Run the topic labeling pipeline
python scripts/update_topic_labels.py
```

### Programmatic Usage

```python
from data_pipeline.processors.topic_labeling import run_topic_labeling_pipeline

# Generate and save updated topic labels
result = run_topic_labeling_pipeline()
print(f"Updated {len(result)} topics")
```

### Service Integration

```python
from service.runtime_glue import RuntimeGlue

# Within the RuntimeGlue service
glue = RuntimeGlue(event_handler=handler, config=config)
glue.update_topic_labels()  # Refresh topic labels
```

## Configuration

### TopicLabeler Parameters

- `min_texts_per_topic` (default: 3) - Minimum texts required for meaningful label generation
- `max_clusters_per_topic` (default: 3) - Maximum subclusters per topic for analysis
- `max_label_words` (default: 3) - Maximum words in generated labels
- `use_embedder` (default: False) - Whether to use DistilBERT embeddings (requires internet)

### File Paths

- **Events Source**: `datasets/events.jsonl` - Input textual examples
- **Topic Lookup**: `datasets/topic_lookup.json` - Output topic mappings
- **Scripts**: `scripts/update_topic_labels.py` - Standalone execution

## Examples

### Before (Placeholder Labels)
```json
{
  "158454": "topic_0",
  "584778": "topic_1", 
  "665808": "topic_2",
  "501754": "viral_tweet_0",
  "341594": "trending_topic_1"
}
```

### After (Meaningful Labels)
```json
{
  "158454": "Youngboy Broke",
  "584778": "Tweet Simulated Number",
  "665808": "Carpenter Tears", 
  "501754": "The Weeknd",
  "341594": "Technology"
}
```

### Dashboard Integration

The updated labels automatically appear in the dashboard:

```
📊 Top-K Live Predictions:
1. Youngboy Broke (Score: 0.95)  
2. Technology (Score: 0.87)
3. Carpenter Tears (Score: 0.82)
```

## Algorithm Details

### Text Collection Strategy
1. **Content Analysis**: Identifies music, technology, gaming, and social media content
2. **Semantic Grouping**: Groups similar content types together
3. **Topic Assignment**: Maps content groups to existing topic IDs deterministically

### Label Generation Process  
1. **Keyword Extraction**: Uses TF-IDF to identify important terms
2. **Pattern Recognition**: Detects artist names, brands, and content types
3. **Label Construction**: Creates concise, meaningful labels from extracted features
4. **Fallback Handling**: Provides generic labels when specific patterns aren't found

### Quality Assurance
- Filters out common stopwords and generic terms
- Limits label length for dashboard compatibility  
- Provides graceful fallbacks for edge cases
- Maintains consistency across similar content types

## Testing

### Unit Tests
```bash
# Run topic labeling tests
pytest tests/unit/test_topic_labeling.py -v
```

### Integration Verification
```bash  
# Verify dashboard integration
python scripts/verify_dashboard_integration.py
```

### Manual Testing
```bash
# Test individual components
python -c "from data_pipeline.processors.topic_labeling import TopicLabeler; labeler = TopicLabeler(use_embedder=False); print('✅ Import successful')"
```

## Dependencies

### Required
- `scikit-learn` - TF-IDF vectorization and clustering
- `numpy` - Numerical operations
- `transformers` - Text processing utilities

### Optional  
- `torch` - For DistilBERT embeddings (when `use_embedder=True`)
- `cachetools` - Embedding caching (when using embedder)

## Performance

- **Execution Time**: ~10-30 seconds for 500+ events
- **Memory Usage**: <100MB for typical workloads  
- **Scalability**: Handles thousands of events efficiently
- **Offline Mode**: Works without internet (TF-IDF only mode)

## Troubleshooting

### Common Issues

**No labels updated**: Check that events file exists and contains text data
```bash
ls -la datasets/events.jsonl
head -n 5 datasets/events.jsonl
```

**Generic labels only**: Increase `min_texts_per_topic` or check text quality
```python
labeler = TopicLabeler(min_texts_per_topic=2)  # Lower threshold
```

**Embedding errors**: Disable embedder for offline mode
```python
run_topic_labeling_pipeline(use_embedder=False)
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

labeler = TopicLabeler()
topic_texts = labeler.collect_texts_by_topic()
for topic_id, texts in topic_texts.items():
    print(f"Topic {topic_id}: {len(texts)} texts")
```

## Architecture

```
Event Data → Text Collection → Clustering → Label Generation → Topic Lookup Update
     ↓              ↓              ↓              ↓                    ↓
events.jsonl → TopicLabeler → TF-IDF/KMeans → Keyword Extract → topic_lookup.json
                                                     ↓
                                              Dashboard Display
```

The pipeline is designed to be:
- **Modular**: Each step can be tested and modified independently
- **Extensible**: New labeling strategies can be easily added
- **Robust**: Graceful handling of missing data and edge cases
- **Integrated**: Seamless compatibility with existing infrastructure