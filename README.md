# Trend Prediction Model & Dashboard

A real-time trend prediction model using Temporal Graph Networks (TGN) to identify emerging trends across social media platforms.

## 🎯 Features

- **Multi-Source Data Collection**: Twitter/X, YouTube, Google Trends
- **Real-time Trend Prediction**: TGN-based emergence detection with Top-K predictions
- **Adaptive Spam Filtering**: Dynamic threshold adjustment for spam resilience
- **SLO Monitoring**: Latency tracking with P50/P95 metrics
- **Live Dashboard**: Streamlit-based monitoring interface
- **Automatic Retraining**: Weekly model updates based on collected data

## 📋 Prerequisites

- Python 3.10 or higher
- Virtual environment tool (venv/conda)
- 4GB+ RAM (8GB recommended)
- Optional: NVIDIA GPU with CUDA 11.8+ for accelerated training

## 🚀 Installation Guide for End Users

### Quick Start (Simple Installation)

1. **Clone the repository**

    ```bash
    git clone <repository-url>
    cd trend-prediction
    ```

2. **Create and activate virtual environment**

    ``` bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies (CPU-only)**

    ```bash
    pip install --upgrade pip
    pip install -e .
    pip install -r requirements-cpu.txt
    ```

4. **Set up API credentials**

    ```bash
    cp .env.example .env
    # Edit .env file and add your API keys:
    # - TWITTER_BEARER_TOKEN (optional)
    # - YOUTUBE_API_KEY (optional)
    ```

5. **Start the service**

    ```bash
    python src/service/main.py
    ```

6. **Open the dashboard** (in a new terminal)

    ```bash
    streamlit run dashboard/app.py
    ```

    Visit <http://localhost:8501> to view the dashboard.

### What You'll See

- **Live Predictions**: Top trending topics with confidence scores
- **Countdown Timers**: Time until predictions are frozen (Δ-freeze)
- **System Health**: Latency metrics and SLO compliance status

## 👩‍💻 Installation Guide for Developers

### Development Setup

1. Clone and setup development environment

    ```bash
    git clone <repository-url>
    cd trend-prediction

    # Create virtual environment
    python3.10 -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    # or
    .venv\Scripts\activate  # Windows
    ```

2. Install in development mode with GPU support (optional)

    ```bash
    pip install --upgrade pip setuptools wheel

    # For CPU-only development
    pip install -e .
    pip install -r requirements-cpu.txt

    # For CUDA 11.8 GPU support
    pip install -r requirements-cuda118.txt --extra-index-url https://download.pytorch.org/whl/cu118
    ```

3. Configure environment

    ```bash
    cp .env.example .env
    ```

    **Edit .env with your API credentials:**

    ```env
    TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
    YOUTUBE_API_KEY=your_youtube_api_key_here
    ```

4. Run tests

    ```bash
    # Unit tests
    pytest tests/unit/

    # Integration tests
    pytest tests/integration/

    # All tests with coverage
    pytest --cov=src tests/
    ```

### Running the Full Pipeline

#### Option 1: Unified Service (Recommended)

```bash
# Start main service with all components
python src/service/main.py

# Optional: provide custom config
python src/service/main.py config/custom.yaml
```

#### Option 2: Component Testing

**Test data collectors individually:**

```bash
# Simulate data collection
python tests/integration/run_ingestion_sim.py
```

**Test preprocessing:**

```bash
# Build TGN edge file from collected events
python -c "from data_pipeline.processors.preprocessing import build_tgn; build_tgn()"
python -m data_pipeline.processors.preprocessing # OR run script directly from module

# Generate meaningful topic labels from textual examples
python scripts/update_topic_labels.py
```

**Test model training:**

```bash
# Run training with noise injection
python src/model/training/train.py

# Hyperparameter tuning
python src/model/training/tune.py
```

### Dashboard Development

```bash
# Run dashboard with hot-reload
streamlit run dashboard/app.py --server.runOnSave true

# Test individual panels
python -m dashboard.components.topk
python -m dashboard.components.latency
python -m dashboard.components.robustness
```

### Project Structure

```text
trend-prediction/
├── src/
│   ├── config/           # Configuration and schemas
│   ├── data_pipeline/    # Data collection & processing
│   │   ├── collectors/   # Twitter, YouTube, Google Trends
│   │   ├── processors/   # Text embedding, preprocessing
│   │   └── storage/      # Graph builder, database
│   ├── model/            # ML components
│   │   ├── core/         # TGN model implementation
│   │   ├── training/     # Training utilities
│   │   └── inference/    # Spam filter, adaptive thresholds
│   ├── service/          # Service orchestration
│   └── utils/            # Helper utilities
├── dashboard/            # Streamlit dashboard
│   ├── components/       # Panel implementations
│   └── layouts/          # UI helpers
├── tests/                # Test suite
├── datasets/             # Data storage (auto-created)
└── logs/                 # Service logs (auto-created)
```

### Key Configuration Files

`src/config/config.py` - Main configuration constants:

- `DELTA_HOURS`: Prediction freeze window (default: 2)
- `WINDOW_MIN`: Evaluation window (default: 60)
- `SLO_MED_MS`: Median latency SLO (default: 1000ms)
- `SLO_P95_MS`: P95 latency SLO (default: 2000ms)

### Development Workflow

1. **Add new data source**:
    - Create collector in `src/data_pipeline/collectors/`
    - Implement `BaseCollector` interface
    - Add to `src/service/main.py` event stream

2. **Modify model architecture**:
    - Edit `src/model/core/tgn.py`
    - Update training in `src/model/training/train.py`
    - Add tests in `tests/unit/`
3. **Add dashboard panel**:
    - Create component in `dashboard/components/`
    - Register in `dashboard/app.py`
    - Use shared layouts from `dashboard/layouts/`

### Monitoring & Debugging

**Check logs:**

```bash
tail -f datasets/*.log
tail -f datasets/metrics_hourly/*.json
```

**Monitor predictions cache:**

```bash
watch -n 5 "cat datasets/predictions_cache.json | python -m json.tool | head -20"
```

**Test adaptive thresholds:**

```bash
python tests/unit/test_adaptive_thresholds.py
```

## 📊 Understanding the System

### Data Flow

1. **Collection**: Real-time events from social platforms
2. **Processing**: Text embedding, spam scoring, timestamp normalization
3. **Graph Building**: Temporal edge stream construction
4. **Model Inference**: TGN forward pass for trend scoring
5. **Evaluation**: Precision@K with Δ-hour label freeze
6. **Monitoring**: Latency tracking, SLO compliance, drift detection

### Key Metrics

- **Precision@K**: Accuracy of Top-K predictions
- **Latency P50/P95**: Service response time percentiles
- **Spam Rate**: Percentage of detected spam events
- **Adaptivity Score**: Model adjustment to distribution shifts

### API Endpoints

The service doesn't expose REST APIs directly but uses an event-driven architecture. Events flow through:

- Collectors → EventHandler → GraphBuilder → TGN Model → Metrics → Dashboard

## 🔧 Troubleshooting

### Common Issues

**No predictions showing:**

- Check if collectors are running: `tail -f datasets/events.jsonl`
- Verify API keys in `.env` file
- Ensure preprocessing has run: `ls datasets/tgn_edges_basic.npz`

**High latency warnings:**

- Reduce batch size in `RealtimeTextEmbedder`
- Switch to CPU if GPU memory is limited
- Check `datasets/tgn_inference.log` for bottlenecks

**Dashboard not updating:**

- Verify predictions cache: `cat datasets/predictions_cache.json`
- Check metrics directory: `ls datasets/metrics_hourly/`
- Restart dashboard: `streamlit run dashboard/app.py`
