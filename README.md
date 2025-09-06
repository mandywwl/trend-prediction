# Trend Prediction System

A real-time trend prediction system that analyzes social media data streams to predict emerging trends using temporal graph neural networks (TGN).

## 🏗️ Architecture Overview

The system consists of several key components:

- **Data Pipeline**: Collects real-time data from Twitter, YouTube, and Google Trends
- **Processing Layer**: Text embedding, spam filtering, and event processing
- **Model Layer**: Temporal Graph Neural Network for trend prediction
- **Dashboard**: Real-time monitoring and visualization
- **Service Layer**: Orchestrates the entire pipeline with metrics and monitoring

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing     │    │     Model       │
│                 │────▶│                  │────▶│                 │
│ • Twitter/X     │    │ • Text Embedding │    │ • TGN Core      │
│ • YouTube       │    │ • Spam Filter    │    │ • Inference     │
│ • Google Trends │    │ • Graph Builder  │    │ • Thresholds    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Dashboard     │◀───│   Service Layer  │◀────────────┘
│                 │    │                  │
│ • Live Metrics  │    │ • Runtime Glue   │
│ • Top-K Trends  │    │ • Event Handler  │
│ • SLO Monitor   │    │ • Metrics Writer │
│ • Robustness    │    │ • Config Mgmt    │
└─────────────────┘    └──────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.10 or 3.11 recommended (3.12 works but may have fewer prebuilt wheels)
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: CPU-only supported, GPU optional for acceleration

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade packaging tools (recommended):

```bash
python -m pip install -U pip setuptools wheel
```

### 2. Install the Package

Install in editable mode using the `pyproject.toml` configuration:

```bash
pip install -e .
```

This registers `src/` as a proper package (`trend-prediction`), enabling clean imports throughout the codebase.

### 3. Install Platform-Specific Dependencies

Choose **one** of the following based on your setup:

#### Option A: CPU-only (macOS, Windows, Linux)

```bash
pip install -r requirements-cpu.txt
```

#### Option B: NVIDIA GPU with CUDA 11.8 (Windows/Linux)

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements-cuda118.txt
```

> **Note**: If you see "No matching distribution found for torch==...+cu118" on macOS, you've selected the wrong option. Use the CPU-only requirements instead.

### 4. Install Browser Dependencies

For web scraping components (TikTok, etc.):

```bash
playwright install
```

### 5. Verify Installation

Check your PyTorch backend:

```bash
python -c "
import torch, platform
print('Python:', platform.python_version())
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available (macOS):', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
"
```

## 🎯 Running the Trend Prediction Model

### Basic Usage

Start the complete trend prediction service:

```bash
python -m service.main
```

This will:
1. Initialize all data collectors (Twitter, YouTube, Google Trends)
2. Start the temporal graph processing pipeline
3. Begin real-time trend prediction
4. Save periodic checkpoints and metrics

### Configuration Options

You can provide a YAML configuration file for custom settings:

```bash
python -m service.main config/custom_config.yaml
```

### Environment Variables

Control preprocessing behavior:

```bash
# Force rebuild of preprocessing data
PREPROCESS_FORCE=1 python -m service.main

# Skip preprocessing entirely
SKIP_PREPROCESS=1 python -m service.main
```

### Data Collection Modes

The system supports both live and simulated data collection:

- **Live Mode**: Uses real APIs (requires valid API keys)
- **Demo Mode**: Uses simulated data streams (default for development)

To use live data, update the API keys in the configuration:

```python
# In service/main.py or via environment variables
TWITTER_BEARER_TOKEN = "your_actual_token"
YOUTUBE_API_KEY = "your_actual_key"
```

## 📊 Dashboard Usage

### Starting the Dashboard

Launch the Streamlit dashboard in a separate terminal:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Dashboard Features

1. **📊 Now (Top-K Live)**: Real-time trending topics and predictions
2. **⏱️ Latency & SLOs**: Performance monitoring and SLA compliance
3. **🛡️ Robustness**: Spam detection and system health metrics
4. **📋 About**: System information and configuration

### Dashboard Components

- **Live Predictions**: Top-K trending topics with confidence scores
- **Performance Metrics**: Latency percentiles, throughput, SLO status
- **Health Monitoring**: Spam rates, adaptive thresholds, alerts
- **Historical Data**: Trend evolution over time

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run specific test categories:

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Tests with coverage
pytest --cov=src tests/
```

### Test Structure

- `tests/unit/`: Component-level tests
- `tests/integration/`: End-to-end pipeline tests
- `tests/fixtures/`: Shared test data and utilities

## 🛠️ Development

### Project Structure

```
trend-prediction/
├── src/                          # Main source code
│   ├── config/                   # Configuration management
│   ├── data_pipeline/           # Data collection and processing
│   │   ├── collectors/          # Data source collectors
│   │   ├── processors/          # Text processing and embedding
│   │   ├── storage/             # Graph building and storage
│   │   └── transformers/        # Data transformation utilities
│   ├── model/                   # Machine learning components
│   │   ├── core/                # TGN implementation
│   │   ├── training/            # Training utilities
│   │   ├── inference/           # Inference and filtering
│   │   └── evaluation/          # Metrics and evaluation
│   ├── service/                 # Service orchestration
│   │   ├── api/                 # API endpoints
│   │   ├── services/            # Core business logic
│   │   └── workers/             # Background tasks
│   └── utils/                   # Shared utilities
├── dashboard/                   # Streamlit dashboard
│   ├── components/             # Dashboard components
│   ├── layouts/                # Layout utilities
│   ├── static/                 # CSS and assets
│   └── utils/                  # Dashboard utilities
├── tests/                      # Test suite
└── docs/                       # Documentation
```

### Key Design Principles

1. **Modularity**: Clear separation of concerns between components
2. **Extensibility**: Easy to add new data sources and models
3. **Observability**: Comprehensive logging and metrics
4. **Reliability**: Graceful error handling and recovery
5. **Performance**: Optimized for real-time processing

### Adding New Data Sources

1. Create a new collector class inheriting from `BaseCollector`
2. Implement the required methods (`start`, `stop`, event processing)
3. Add the collector to the main service configuration
4. Update the dashboard to display metrics from the new source

Example:

```python
from data_pipeline.collectors.base import BaseCollector

class MyCollector(BaseCollector):
    def __init__(self, on_event=None):
        super().__init__("my_source", on_event)
    
    def start(self):
        # Implementation here
        pass
    
    def stop(self):
        # Implementation here  
        pass
```

## 📈 Performance Tuning

### Key Performance Metrics

- **Latency**: End-to-end processing time per event
- **Throughput**: Events processed per second
- **Memory Usage**: Graph size and embedding cache
- **Accuracy**: Prediction quality metrics

### Optimization Strategies

1. **Batch Processing**: Adjust batch sizes for embedding and inference
2. **Caching**: Optimize node feature and embedding caches
3. **Pruning**: Remove old graph edges based on time decay
4. **Parallelization**: Use multiple collector threads

### Configuration Tuning

Key parameters to adjust in configuration:

```yaml
# Runtime configuration
embedding:
  batch_size: 8
  max_latency_ms: 50
  device: "cpu"  # or "cuda" for GPU

graph:
  max_nodes: 10000
  time_decay: 0.95
  
monitoring:
  checkpoint_interval: 100
  metrics_interval: 3600
```

## 🚨 Monitoring and Alerts

### Built-in Monitoring

The system includes comprehensive monitoring:

- **SLO Tracking**: Latency percentiles and compliance
- **Health Checks**: Component status and error rates  
- **Resource Usage**: Memory, CPU, and disk utilization
- **Data Quality**: Spam rates and validation metrics

### Setting Up Alerts

Configure alert thresholds in the dashboard or via configuration:

```yaml
alerts:
  latency_p95_ms: 1000
  spam_rate_threshold: 0.1
  error_rate_threshold: 0.05
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is installed in editable mode (`pip install -e .`)
2. **Memory Issues**: Reduce batch sizes or enable graph pruning
3. **API Rate Limits**: Check collector error logs and implement backoff
4. **Missing Dependencies**: Verify all requirements files are installed

### Debug Mode

Enable verbose logging:

```bash
# Set logging level
export LOG_LEVEL=DEBUG
python -m service.main
```

### Log Locations

- **Service Logs**: Console output or configured log files
- **Dashboard Logs**: Streamlit console output
- **Metrics**: `datasets/metrics/` directory
- **Checkpoints**: `datasets/checkpoints/` directory

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

[Add support contact information here]