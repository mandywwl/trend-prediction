# Project Structure Summary

## Structure Layers 

### Service Layer (`src/service/`)
```
src/service/
├── main.py                    # Main entry point
├── api/                      # API layer components
├── services/                 # Service layer
│   ├── inference/           # Inference services
│   │   └── tgn_service.py   # TGN inference service
│   ├── preprocessing/       # Preprocessing services  
│   │   └── event_handler.py # Event handler
│   ├── data_collection/     # Data collection services
│   └── monitoring/          # Monitoring services
└── workers/                 # Background workers
```

### Data Pipeline (`src/data_pipeline/`)
```
src/data_pipeline/
├── collectors/              # Data collectors
│   ├── twitter_collector.py
│   ├── youtube_collector.py
│   └── google_trends_collector.py
├── processors/              # Data processors
│   ├── preprocessing.py
│   └── text_rt_distilbert.py
├── transformers/            # Data transformers
│   ├── event_parser.py
│   └── validation.py
└── storage/                 # Storage utilities
    ├── builder.py
    └── decay.py
```

### Model Components (`src/model/`)
```
src/model/
├── core/                    # Core model components
│   ├── tgn.py
│   └── losses.py
├── training/                # Training components
│   ├── train.py
│   ├── train_growth.py
│   └── noise_injection.py
├── inference/               # Inference components
│   ├── spam_filter.py
│   └── adaptive_thresholds.py
└── evaluation/              # Evaluation components
    └── metrics.py
```

### Configuration (`src/config/`)
```
src/config/
├── base.py                  # Base configuration
├── models.py                # Configuration models
├── schemas.py               # Data schemas (existing)
└── environments/            # Environment-specific configs
    ├── development.py
    ├── production.py
    └── testing.py
```

### Utilities (`src/utils/`)
```
src/utils/
├── decorators.py            # Utility decorators
├── validation.py            # Validation utilities
├── datetime.py              # DateTime utilities
├── monitoring.py            # Monitoring utilities
├── io.py                    # IO utilities (existing)
├── logging.py               # Logging utilities (existing)
└── time.py                  # Time utilities (existing)
```

### Dashboard (`dashboard/`)
```
dashboard/
├── components/              # Dashboard components (existing)
├── layouts/                 # Layout components
├── utils/                   # Dashboard utilities
├── static/                  # Static assets
└── templates/               # Templates
```

### Testing (`tests/`)
```
tests/
├── conftest.py             # Test configuration
├── unit/                   # Unit tests
├── integration/            # Integration tests
├── fixtures/               # Test fixtures
└── helpers/                # Test helpers
```

### Documentation (`docs/`)
```
docs/
├── api/                    # API documentation
├── architecture/           # Architecture documentation
├── deployment/             # Deployment documentation
├── development/            # Development documentation
└── user/                   # User documentation
```