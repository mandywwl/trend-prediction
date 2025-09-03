import time
import numpy as np

from features.text_rt_distilbert import RealtimeTextEmbedder
from config.config import EMBEDDER_P95_CPU_MS


def test_realtime_embedder_cache_and_latency():
    embedder = RealtimeTextEmbedder(
        batch_size=2, max_latency_ms=10, device="cpu", cache_ttl=10
    )

    # Cold encode
    start = time.perf_counter()
    emb1 = embedder.encode(["hello world"])
    cold = time.perf_counter() - start

    # Cached encode
    start = time.perf_counter()
    emb2 = embedder.encode(["hello world"])
    warm = time.perf_counter() - start

    assert np.allclose(emb1, emb2, atol=1e-6)
    assert warm * 3 < cold

    # Latency check (p95 below configured CPU target for short texts)
    embedder.encode(["warmup"])  # warm-up
    durations = []
    for i in range(20):
        t = f"text {i}"
        start = time.perf_counter()
        embedder.encode([t])
        durations.append(time.perf_counter() - start)
    p95 = np.percentile(durations, 95) * 1000
    assert p95 < EMBEDDER_P95_CPU_MS
