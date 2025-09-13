# Trend Prediction Model & Dashboard

A real-time trend prediction model using Temporal Graph Networks (TGN) to identify emerging trends across social media platforms.

## 🎯 Features

- **Multi-Source Data Collection**: Twitter/X, YouTube, Google Trends
- **TGN-Based Prediction**: Temporal Graph Network model for growth score prediction
- **Adaptive System**:
  - Dynamic spam filtering with edge weight adjustment
  - Adaptive threshold control based on spam rates
  - Back-pressure mechanisms for latency management
- **Automated Pipeline**:
  - Periodic model retraining (configurable, default: weekly)
  - Topic labeling pipeline for meaningful trend names
  - Database storage with SQLite backend
- **Live Dashboard**: Streamlit-based monitoring with:
  - Top-K predictions with confidence scores
  - Δ-hour label freeze for evaluation
  - Latency tracking (P50/P95) with SLO monitoring
  - Robustness metrics and spam rate visualization
- **Production Features**:
  - Hot-reload of model checkpoints
  - Graceful shutdown handling
  - Event logging (JSONL + database)
  - Configurable runtime parameters via YAML
- **SLO Monitoring**: Latency tracking with P50/P95 metrics
- **Live Dashboard**: Streamlit-based monitoring interface
- **Automatic Retraining**: Weekly model updates based on collected data

## 📋 Prerequisites

- Python 3.10 or higher
- Virtual environment tool (venv/conda)
- 4GB+ RAM (8GB recommended)
- Optional: NVIDIA GPU with CUDA 11.8+ for accelerated training

## 🚀 Quick Start Installation

1. **Clone and Setup Environment**

    ```bash
    git clone <repository-url>
    cd trend-prediction

    # Create and activate virtual environment
    python -m venv .venv

    # Windows
    .venv\Scripts\activate

    # Linux/Mac
    source .venv/bin/activate
    ```

2. **Install dependencies**

    ```bash
    pip install --upgrade pip
    pip install -e .

    # For CPU-only installation
    pip install -r requirements-cpu.txt

    # For GPU support
    pip install -r requirements-cuda118.txt --extra-index-url https://download.pytorch.org/whl/cu118
    ```

3. **Configure API credentials**

    ```bash
    cp .env.example .env
    # Edit .env and add your API keys:
    # TWITTER_BEARER_TOKEN=your_token_here  (optional - uses simulation if not provided)
    # YOUTUBE_API_KEY=your_key_here
    ```

4. **Start the service**

    ```bash
    python src/service/main.py

    # Optional: provide custom configuration
    python src/service/main.py config/custom_runtime_config.yaml
    ```

5. **Open the dashboard** (in a new terminal)

    ```bash
    streamlit run dashboard/app.py
    ```

    Visit <http://localhost:8501> to view the dashboard.

## 📊 System Architecture

### Data Flow Pipeline

1. **Collection**: Multi-source event ingestion with simulated fallbacks
2. **Preprocessing**: Real-time text embedding using DistilBERT
3. **Spam Scoring**: Heuristic-based spam detection and edge weighting
4. **Graph Building**: Temporal graph construction with LRU node management
5. **TGN Inference**: Growth score prediction via temporal memory networks
6. **Evaluation**: Online Precision@K with Δ-hour label freeze
7. **Monitoring**: Real-time metrics, latency tracking, and dashboard updates

### Key Components

- **EventHandler**: Orchestrates preprocessing, spam filtering, and TGN updates
- **RuntimeGlue**: Manages streaming loop, metrics computation, and cache updates
- **TGNInferenceService**: Wraps trained TGN model for online inference
- **SensitivityController**: Adaptive threshold management based on spam rates
- **TrainingScheduler**: Automated periodic retraining with database integration

## 🔧 Configuration

### Runtime Configuration (YAML)

Create a runtime_config.yaml:

```yaml
runtime:
  update_interval_sec: 60          # Dashboard update frequency
  enable_background_timer: true    # Enable continuous updates
  delta_hours: 2                   # Prediction freeze window
  window_min: 60                   # Rolling window in minutes
  k_default: 5                     # Default Top-K predictions
  k_options: [3, 5, 10, 20]       # Available K values
```

### Key Configuration Parameters (src/config/config.py)

- **Model Settings:**
  - `TGN_MEMORY_DIM`: 100 (memory dimension)
  - `TGN_TIME_DIM`: 10 (time encoding dimension)
  - `TRAIN_EPOCHS`: 8 (training epochs)

- **Evaluation:**
  - `DELTA_HOURS`: 2 (label freeze window)
  - `WINDOW_MIN`: 60 (evaluation window)

- **Performance:**
  - `SLO_MED_MS`: 1000 (median latency SLO)
  - `SLO_P95_MS`: 2000 (P95 latency SLO)

- **Training:**
  - `TRAINING_INTERVAL_HOURS`: 168 (weekly)
  - `MIN_EVENTS_FOR_TRAINING`: 100

## 👩‍💻 Development Guide

### Running Tests

```bash
# Unit tests
pytest tests/unit/

#  Integration tests  
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/
```

### Training the Model

```bash
# Preprocess events to create TGN edge file
python -m data_pipeline.processors.preprocessing

# Train TGN model
python src/model/training/train.py

# Hyperparameter tuning
python src/model/training/tune.py
```

### Working with Components

#### Test data collection only

```bash
python tests/integration/run_ingestion_sim.py
```

#### Update topic labels

```bash
python scripts/update_topic_labels.py
```

#### Evaluate model performance

```bash
python -m model.evaluation.baseline_eval --events data/events.jsonl --outdir data/eval_predictive
```

### Project Structure

```text
trend-prediction/
├── src/
│   ├── config/               # Configuration and schemas
│   ├── data_pipeline/        # Data collection & processing
│   │   ├── collectors/       # Twitter, YouTube, Google Trends
│   │   ├── processors/       # Embedding, preprocessing, labeling
│   │   └── storage/         # Graph builder, database, decay
│   ├── model/               # ML components
│   │   ├── core/           # TGN model, losses
│   │   ├── training/       # Training, noise injection, tuning
│   │   ├── inference/      # Spam filter, adaptive thresholds
│   │   └── evaluation/     # Metrics, baseline evaluation
│   ├── service/            # Service orchestration
│   │   ├── main.py        # Main entry point
│   │   ├── runtime_glue.py # Stream processing & metrics
│   │   ├── tgn_service.py  # TGN inference wrapper
│   │   └── training_scheduler.py # Automated retraining
│   └── utils/              # Helper utilities
├── dashboard/              # Streamlit dashboard
├── tests/                  # Test suite
├── data/                  # Data storage (auto-created)
│   ├── events.jsonl       # Raw event stream
│   ├── events.db          # SQLite database
│   ├── tgn_edges_basic.npz # Preprocessed graph data
│   ├── tgn_model.pt       # Trained model checkpoint
│   ├── predictions_cache.json # Dashboard cache
│   └── metrics_hourly/    # Hourly metrics snapshots
```

## 📊 Understanding the Metrics

### Precision@K

- Measures accuracy of Top-K predictions
- Uses Δ-hour label freeze for fair evaluation
- Computed over rolling window (default: 60 minutes)

### Latency Tracking

- **P50/P95**: Median and 95th percentile latencies
- **Per-stage breakdown**: Ingest, preprocess, model, postprocess
- **SLO compliance**: Visual indicators for target thresholds

### Robustness Metrics

- **Spam Rate**: Percentage of detected spam events
- **Downweighted %**: Edges with reduced influence
- **θ_g, θ_u**: Adaptive threshold values

### Adaptivity Score

- Measures model's ability to adjust to distribution shifts
- Based on MAPE (Mean Absolute Percentage Error) decay

## 🔧 Troubleshooting

### Common Issues

### **No predictions showing:**

- Check collectors are running: `tail -f data/events.jsonl`
- Verify preprocessing: `ls -la data/tgn_edges_basic.npz`
- Check model checkpoint: `ls -la data/tgn_model.pt`

### **High latency warnings:**

- Reduce `EMBEDDER_BATCH_SIZE` in config
- Check GPU memory if using CUDA
- Monitor `data/tgn_inference.log`

### **Dashboard not updating:**

- Check background timer is enabled in config
- Verify predictions cache: `cat data/predictions_cache.json`
- Check metrics: `ls data/metrics_hourly/`
- Restart dashboard if needed

### **Database issues:**

- Check database file: `ls -la data/events.db`
- Verify write permissions in data directory
- Check disk space availability

## 🚦 Monitoring & Operations

### **Log Files**

- `data/events.jsonl`: Raw event stream
- `data/tgn_inference.log`: Model latency tracking
- `data/emergence_labels.log`: Label decisions
- `data/adaptive_thresholds.log`: Threshold adjustments

### Health Checks

```bash
# Check event ingestion rate
tail -f data/events.jsonl | grep -c "timestamp"

# Monitor model inference
tail -f data/tgn_inference.log

# Watch predictions cache updates
watch -n 5 "cat data/predictions_cache.json | python -m json.tool | head -20"
```

### Performance Tuning

- Adjust `EMBEDDER_BATCH_SIZE` for throughput/latency trade-off
- Tune `SPAM_WINDOW_MIN` for spam detection sensitivity
- Modify `THRESH_RAISE_FACTOR` for adaptive threshold aggressiveness
- Set `TRAINING_INTERVAL_HOURS` based on data velocity

## 🎓 Technical Details

### TGN Model

- Temporal Graph Network with memory module
- Growth score prediction via regression head
- Online memory updates with LRU eviction
- Configurable memory and time dimensions

### Adaptive Mechanisms

- Spam-aware edge weighting (half-life decay + spam score)
- Dynamic threshold adjustment based on spam rates
- Back-pressure control for latency management
- EMA calibration for stable predictions

### Data Sources

- Twitter/X: Real API or realistic simulation
- TikTok — using Playwright + ViT + CLIP: *(TBA)*
- YouTube: API integration with trending videos
- Google Trends: Real API *(TBA)* or Time-aware trend simulation

## 📃 Documentation
