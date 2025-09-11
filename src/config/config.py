"""Project-wide configuration constants."""

from pathlib import Path
from utils.path_utils import find_repo_root

# Base directory configuration
PROJECT_ROOT = find_repo_root()
DATA_DIR = PROJECT_ROOT / "datasets"
LOGS_DIR = PROJECT_ROOT / "logs"

# TGN model defaults
TGN_MEMORY_DIM: int = 100
TGN_TIME_DIM: int = 10
TGN_EDGE_DIM: int = 768  # Default to DistilBERT hidden size
INFERENCE_DEVICE: str = "cpu"  # "cpu" or "cuda"
MAX_NODES: int = 1_000_000  # example cap; keep current behaviour if not used

# Text embedding policy for missing text features
DEFAULT_TEXT_EMB_POLICY: str = "zeros"  # "zeros" | "mean" | "learned_unknown"

# Service configuration
DEFAULT_EMBED_BUDGET_MS = 50

# Online evaluation
DELTA_HOURS: int = 2  # Δ
WINDOW_MIN: int = 60  # W
K_DEFAULT: int = 5
K_OPTIONS: tuple[int, ...] = (5, 10)

# SLOs
SLO_MED_MS: int = 1000
SLO_P95_MS: int = 2000

# Embeddings / preprocessing budgets
EMBEDDER_P95_CPU_MS: int = 100
EMBED_PREPROC_BUDGET_MS: int = 200  # additional budget in Story 2

# Robustness & sensitivity
SPAM_WINDOW_MIN: int = 60
SPAM_RATE_SPIKE: float = 0.20  # >=20% considered spike
THRESH_RAISE_FACTOR: float = 1.20  # +20% during spike
THRESH_DECAY_RATE: float = 0.9  # multiplicative decay per window
EDGE_WEIGHT_MIN: float = 0.2  # clamp for down-weighting

# Synthetic noise injection
NOISE_P_DEFAULT: float = 0.08
NOISE_JITTER_MIN_MIN: float = 1.0
NOISE_JITTER_MAX_MIN: float = 15.0

# Training objectives
HUBER_DELTA_DEFAULT: float = 1.0
LABEL_SMOOTH_EPS: float = 0.05


# IO paths (keep existing if already defined elsewhere)
METRICS_SNAPSHOT_DIR: str = "datasets/metrics_hourly"
PREDICTIONS_CACHE_PATH: str = "datasets/predictions_cache.json"
TOPIC_LOOKUP_PATH: str = "datasets/topic_lookup.json"
METRICS_LOOKUP_PATH: str = "datasets/metrics_lookup.json"

# Regions
REGION_DEFAULTS: tuple[str, ...] = ("SG",)

# Timezone policy for bucketing
BUCKET_TZ: str = "UTC"

# Realtime embedding service guards
EMBED_MAX_BACKLOG: int = 32
EMBED_MAX_TOKENS: int = 32


