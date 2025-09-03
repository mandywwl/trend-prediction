from __future__ import annotations

"""Project-wide configuration constants."""

# Online evaluation
DELTA_HOURS: int = 2         # Δ
WINDOW_MIN: int = 60         # W
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
SPAM_RATE_SPIKE: float = 0.20       # >=20% considered spike
THRESH_RAISE_FACTOR: float = 1.20   # +20% during spike
THRESH_DECAY_RATE: float = 0.9      # multiplicative decay per window
EDGE_WEIGHT_MIN: float = 0.2        # clamp for down-weighting

# Synthetic noise injection
NOISE_P_DEFAULT: float = 0.08
NOISE_JITTER_MIN_MIN: int = 3
NOISE_JITTER_MAX_MIN: int = 10

# Training objectives
HUBER_DELTA_DEFAULT: float = 1.0
LABEL_SMOOTH_EPS: float = 0.05

# Limits / guards
MAX_NODES: int = 1_000_000  # example cap; keep current behaviour if not used
DEFAULT_TEXT_EMB_POLICY: str = "zeros"  # "zeros" | "mean" | "learned_unknown"

# IO paths (keep existing if already defined elsewhere)
METRICS_SNAPSHOT_DIR: str = "datasets/metrics_hourly"
PREDICTIONS_CACHE_PATH: str = "datasets/predictions_cache.json"

# Regions
REGION_DEFAULTS: tuple[str, ...] = ("SG",)

# Timezone policy for bucketing
BUCKET_TZ: str = "UTC"

# Realtime embedding service guards
EMBED_MAX_BACKLOG: int = 32
EMBED_MAX_TOKENS: int = 32
