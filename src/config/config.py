"""Project-wide single-source configuration constants for trend prediction service."""

from pathlib import Path
from utils.path_utils import find_repo_root

# ----- Base directory configuration -------
PROJECT_ROOT = find_repo_root()
DATA_DIR: Path = PROJECT_ROOT / "datasets"

# ------ IO paths (keep existing if already defined elsewhere) -------
METRICS_SNAPSHOT_DIR: Path = DATA_DIR / "metrics_hourly"
PREDICTIONS_CACHE_PATH: Path = DATA_DIR / "predictions_cache.json"
TOPIC_LOOKUP_PATH: Path = DATA_DIR / "topic_lookup.json"
METRICS_LOOKUP_PATH: Path = DATA_DIR / "metrics_lookup.json"
DATABASE_PATH: Path = DATA_DIR / "events.db"
EVENT_JSONL_PATH: Path = DATA_DIR / "events.jsonl"
EVENT_MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB

# ------ Device configuration -------
INFERENCE_DEVICE: str = "cpu"  # "cpu" or "cuda"

# ------ TGN model defaults -------
TGN_MEMORY_DIM: int = 100        # Memory dimension
TGN_TIME_DIM: int = 10           # Time encoding dimension
TGN_EDGE_DIM: int = 768          # Default to DistilBERT hidden size
TGN_LR: float = 0.001            # Learning rate
TGN_DECODER_HIDDEN: int = 100    # Hidden layer size in the decoder MLP
MAX_NODES: int = 1_000_000       # Max nodes in TGN graph

# ------- Topic configuration -------
TOPIC_REFRESH_EVERY: int = 100   # Refresh topics every N events
TOPIC_REFRESH_SECS: int = 600    # 10 minutes in seconds; refresh interval

# ------- Embedding configuration -------
TEXT_EMB_POLICY: str = "zeros"      # "zeros" | "mean" | "learned_unknown"
EMBEDDER_BATCH_SIZE: int = 8        # max batch size for embedder
EMBEDDER_MAX_LATENCY_MS: int = 50   # max wait time to form
EMBEDDER_P95_CPU_MS: int = 100      # p95 CPU time per batch
EMBED_PREPROC_BUDGET_MS: int = 200  # additional budget in Story 2
EMBED_MAX_TOKENS: int = 32          # max tokens for DistilBERT
EMBED_MAX_BACKLOG: int = 32         # max queued requests

# ------- Collector/stream configuration -------
KEYWORDS = ["#trending", "fyp", "viral"]
DEFAULT_TRENDING_TOPICS = [
                    "artificial intelligence", "climate change", "electric vehicles",
                    "streaming services", "remote work", "social media", "gaming",
                    "cryptocurrency", "space exploration", "renewable energy"
                ]                        # Default topics if none found
TWITTER_SIM_EVENTS_PER_BATCH: int = 4    # Number of simulated events per batch
TWITTER_SIM_BATCH_INTERVAL: int = 120    # 2 minutes in seconds; Twitter API rate limit
TRENDS_CATEGORY: str = "all"             # Google trends category; "all" for all categories
TRENDS_COUNT: int = 8                    # Number of trending topics to fetch
TRENDS_INTERVAL_SEC: int = 1800          # 30 minutes in seconds; Google trends refresh interval
REGIONS: tuple[str, ...] = ("US", "SG",) # Regions for collectors; Currently, collectors only use first region as default

# ------ Training configuration -------
TRAINING_INTERVAL_HOURS: int = 168   # 7 days in hours; Weekly training interval (default)
MIN_EVENTS_FOR_TRAINING: int = 100   # Minimum events to trigger training
SCHEDULER_POLL_SECONDS: int = 3600   # 1 hour in seconds; Scheduler polls every hour (default)
TRAIN_EPOCHS: int = 8                # Number of epochs for TGN training

# Training objectives
HUBER_DELTA_DEFAULT: float = 1.0   # Huber loss delta default
LABEL_SMOOTH_EPS: float = 0.05     # Smoothing for classification labels

# ------- Preprocessing configuration -------
PERIODIC_REBUILD_SECS: int = 604800  # 1 week rebuild of events DB

# ------ Online evaluation -------
DELTA_HOURS: int = 2                 # Δ
WINDOW_MIN: int = 60                 # W
K_DEFAULT: int = 5                   # metrics Prec@K default
K_OPTIONS: tuple[int, ...] = (5, 10) # metrics Prec@K options

# ------- Exponential factor bases for metrics normalization -------
GROWTH_FACTOR_BASE: float = 2.0     # Base for exponential growth factor
UNIQUE_USERS_BASE: int = 50         # Base for exponential unique users factor

# ------ SLOs -------
SLO_MED_MS: int = 1000              # Latency service level objectives median in milliseconds.
SLO_P95_MS: int = 2000              # Latency service level objectives p95 in milliseconds.

# ------- Robustness & sensitivity -------
SPAM_WINDOW_MIN: int = 60           # 1 hour window
SPAM_RATE_SPIKE: float = 0.20       # >=20% considered spike
THRESH_RAISE_FACTOR: float = 1.20   # +20% during spike
THRESH_DECAY_RATE: float = 0.9      # multiplicative decay per window
EDGE_WEIGHT_MIN: float = 0.2        # clamp for down-weighting

# ---------- Synthetic noise injection ----------
NOISE_P_DEFAULT: float = 0.08       # probability of noise event
NOISE_JITTER_MIN_MIN: float = 1.0   # min jitter in minutes
NOISE_JITTER_MAX_MIN: float = 15.0  # max jitter in minutes

# -------- Timezone policy for bucketing -------
BUCKET_TZ: str = "UTC"         # e.g. "UTC", "America/New_York"

# --------- Predictive target config -------
GROWTH_HORIZON_H = 24          # k: predict growth over next 24h
LABEL_WINDOW_H = 1             # smoothing window for counts (rolling)
LABEL_TYPE = "logdiff"         # one of {"diff", "pct", "logdiff"}
LABEL_EPS = 1.0                # epsilon to stabilise logs/ratios

