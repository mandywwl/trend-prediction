"""Schema definitions for structured data used in the project."""

from typing import TypedDict, Dict, List
import numpy as np


class Features(TypedDict, total=False):
    text_emb: np.ndarray  # float32, mean pooled
    # future: image_emb, lang_id, etc.


class Event(TypedDict):
    event_id: str
    ts_iso: str  # timezone-aware ISO8601 (UTC)
    actor_id: str
    target_ids: List[str]
    edge_type: str
    features: Features


Batch = List[Event]


class StageMs(TypedDict):
    ingest: int
    preprocess: int
    model_update_forward: int
    postprocess: int


class LatencySummary(TypedDict):
    median_ms: int
    p95_ms: int
    per_stage_ms: StageMs


class PrecisionAtKSnapshot(TypedDict):
    k5: float
    k10: float
    support: int


class HourlyMetrics(TypedDict):
    precision_at_k: PrecisionAtKSnapshot
    adaptivity: float
    latency: LatencySummary
    meta: Dict[str, str]  # {"generated_at": ISO8601}


class CacheItem(TypedDict):
    t_iso: str
    topk: List[Dict[str, float]]  # {"topic_id": int, "score": float}


class PredictionsCache(TypedDict):
    last_updated: str
    items: List[CacheItem]
