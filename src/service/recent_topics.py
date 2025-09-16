import math
import time
from collections import OrderedDict

class RecentTopicReservoir:
    """
    Keeps recent topic scores with EMA (+ recency decay) and TTL,
    so we can emit >1 candidate per tick no matter how sparse the stream is.
    """
    def __init__(self, decay: float = 0.6, ttl_sec: int = 3600,
                 tau_recency: int = 900, max_size: int = 4096):
        self.decay = float(decay)
        self.ttl = int(ttl_sec)
        self.tau = int(tau_recency)
        self.max_size = int(max_size)
        self._store: dict[int, dict] = {}
        self._lru = OrderedDict()

    def _now(self) -> float:
        return time.time()

    def update(self, tid: int, score: float, ts: float | None = None) -> None:
        ts = self._now() if ts is None else float(ts)
        e = self._store.get(tid)
        if e is None:
            self._store[tid] = {"ema": float(score), "last": ts}
        else:
            e["ema"] = self.decay * float(score) + (1.0 - self.decay) * float(e["ema"])
            e["last"] = ts

        # LRU book-keeping + bound
        if tid in self._lru:
            self._lru.move_to_end(tid)
        else:
            self._lru[tid] = None
            if len(self._lru) > self.max_size:
                old_tid, _ = self._lru.popitem(last=False)
                self._store.pop(old_tid, None)

    def top_n(self, n: int = 5) -> list[tuple[int, float]]:
        now = self._now()
        out = []
        for tid, e in list(self._store.items()):
            if now - e["last"] > self.ttl:
                self._store.pop(tid, None)
                self._lru.pop(tid, None)
                continue
            recency = math.exp(-(now - e["last"]) / max(1, self.tau))
            score = float(e["ema"]) * recency
            out.append((tid, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:max(1, int(n))]
