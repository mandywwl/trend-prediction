"""Trend emergence labelling and online Precision@K.

Reads ``DELTA_HOURS`` and ``WINDOW_MIN`` from :mod:`config.config`. Fetches
``(theta_g, theta_u)`` from :class:`model.adaptive_thresholds.SensitivityController`,
applies them to scale baseline thresholds for growth and unique users, and
logs applied values per decision for replay. Replay validates that the same
config values are used and recomputes labels exactly.

Adds ``PrecisionAtKOnline`` which computes online Precision@K using a Δ-hour
label freeze and a rolling hourly snapshot window.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterator, Optional, Tuple, Iterable, List
from collections import deque
import json
import os

from config.config import (
    DELTA_HOURS,
    WINDOW_MIN,
    K_DEFAULT,
    K_OPTIONS,
    METRICS_SNAPSHOT_DIR,
)
from config.schemas import PrecisionAtKSnapshot
from model.inference.adaptive_thresholds import SensitivityController


@dataclass
class EmergenceDecision:
    ts: str
    theta_g: float
    theta_u: float
    delta_hours: int
    window_min: int
    growth_factor_base: float
    unique_users_base: int
    growth_factor_threshold: float
    unique_users_threshold: float
    mentions_curr: int
    mentions_past: int
    unique_users_curr: int
    label: int


class EmergenceLabelBuffer:
    """Sliding-window buffer to compute emergence labels with adaptive thresholds.

    Baselines:
      - growth factor base = 2.0 (mentions must be >= 2x past window)
      - unique users base = 50 (must be >= 50 unique users in current window)

    The SensitivityController provides (theta_g, theta_u) that scale these
    baselines under spam pressure. All decisions are logged for replay.
    """

    def __init__(
        self,
        sensitivity: Optional[SensitivityController] = None,
        *,
        growth_factor_base: float = 2.0,
        unique_users_base: int = 50,
        delta_hours: Optional[int] = None,
        window_min: Optional[int] = None,
        log_path: str = os.path.join("datasets", "emergence_labels.log"),
    ) -> None:
        # Default controller at baseline so thresholds match spec when not provided
        self.sensitivity = sensitivity or SensitivityController()
        self.growth_factor_base = float(growth_factor_base)
        self.unique_users_base = int(unique_users_base)
        self.delta_hours = int(DELTA_HOURS if delta_hours is None else delta_hours)
        self.window_min = int(WINDOW_MIN if window_min is None else window_min)
        self.log_path = log_path

        self._events: Deque[Tuple[datetime, str]] = deque()
        self._emergence_times: Deque[datetime] = deque()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # ------------------------------------------------------------------
    def add_event(self, *, ts_iso: str, user_id: str) -> int:
        """Add an event and return current emergence label (0/1).

        Logs the applied thresholds, counts and label with timestamp.
        """
        now = datetime.fromisoformat(ts_iso)
        self._events.append((now, user_id))
        self._evict_older_than(now - timedelta(minutes=self.window_min))

        mentions_curr, unique_curr = self._counts_in_range(
            start=now - timedelta(minutes=self.window_min), end=now
        )
        past_start = now - timedelta(hours=self.delta_hours + self.window_min)
        past_end = now - timedelta(hours=self.delta_hours)
        mentions_past, _ = self._counts_in_range(start=past_start, end=past_end)

        th = self.sensitivity.thresholds()
        gf_thresh = self.growth_factor_base * th.theta_g
        uu_thresh = int(round(self.unique_users_base * th.theta_u))

        label = int(
            (mentions_curr >= gf_thresh * max(1, mentions_past))
            and (unique_curr >= uu_thresh)
        )
        if label == 1:
            self._emergence_times.append(now)

        self._log(
            EmergenceDecision(
                ts=now.isoformat(timespec="seconds"),
                theta_g=th.theta_g,
                theta_u=th.theta_u,
                delta_hours=self.delta_hours,
                window_min=self.window_min,
                growth_factor_base=self.growth_factor_base,
                unique_users_base=self.unique_users_base,
                growth_factor_threshold=gf_thresh,
                unique_users_threshold=float(uu_thresh),
                mentions_curr=mentions_curr,
                mentions_past=mentions_past,
                unique_users_curr=unique_curr,
                label=label,
            )
        )

        return label

    # ------------------------------------------------------------------
    def emergent_within(self, *, t: datetime, delta_hours: int) -> bool:
        """Return True if any emergence (label=1) occurred in (t, t+Δ].

        Uses the in-memory record of emergence timestamps. Older entries are
        evicted lazily based on the query time to keep memory bounded.
        """
        start = t
        end = t + timedelta(hours=delta_hours)
        cutoff = t - timedelta(hours=self.delta_hours)
        while self._emergence_times and self._emergence_times[0] < cutoff:
            self._emergence_times.popleft()
        for ts in self._emergence_times:
            if start < ts <= end:
                return True
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def replay_from_log(path: str) -> Iterator[EmergenceDecision]:
        """Yield decisions from log and validate recomputed labels.

        Ensures reproduction by recomputing the label from recorded fields
        and raising if any mismatch is detected.
        """
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec: Dict[str, object] = json.loads(line)
                # Back-compat: allow different field names if needed
                ed = EmergenceDecision(
                    ts=str(rec["ts"]),
                    theta_g=float(rec["theta_g"]),
                    theta_u=float(rec["theta_u"]),
                    delta_hours=int(rec["delta_hours"]),
                    window_min=int(rec["window_min"]),
                    growth_factor_base=float(rec["growth_factor_base"]),
                    unique_users_base=int(rec["unique_users_base"]),
                    growth_factor_threshold=float(rec["growth_factor_threshold"]),
                    unique_users_threshold=float(rec["unique_users_threshold"]),
                    mentions_curr=int(rec["mentions_curr"]),
                    mentions_past=int(rec["mentions_past"]),
                    unique_users_curr=int(rec["unique_users_curr"]),
                    label=int(rec["label"]),
                )

                # Ensure config constants match for reproducibility
                if ed.delta_hours != DELTA_HOURS or ed.window_min != WINDOW_MIN:
                    raise ValueError(
                        "Replay mismatch: DELTA_HOURS/WINDOW_MIN differ from config",
                    )

                # Recompute label using the same thresholds and counts
                gf = ed.growth_factor_threshold
                uu = ed.unique_users_threshold
                recomputed = int(
                    (ed.mentions_curr >= gf * max(1, ed.mentions_past))
                    and (ed.unique_users_curr >= uu)
                )
                if recomputed != ed.label:
                    raise ValueError(
                        "Replay mismatch: recomputed label differs from logged label"
                    )
                yield ed

    # ----------------------- Internals --------------------------------
    def _evict_older_than(self, cutoff: datetime) -> None:
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _counts_in_range(self, *, start: datetime, end: datetime) -> Tuple[int, int]:
        mentions = 0
        users: set[str] = set()
        for ts, uid in self._events:
            if start <= ts <= end:
                mentions += 1
                users.add(uid)
        return mentions, len(users)

    def _log(self, decision: EmergenceDecision) -> None:
        record = {
            "ts": decision.ts,
            "theta_g": decision.theta_g,
            "theta_u": decision.theta_u,
            "delta_hours": decision.delta_hours,
            "window_min": decision.window_min,
            "growth_factor_base": decision.growth_factor_base,
            "unique_users_base": decision.unique_users_base,
            "growth_factor_threshold": decision.growth_factor_threshold,
            "unique_users_threshold": decision.unique_users_threshold,
            "mentions_curr": decision.mentions_curr,
            "mentions_past": decision.mentions_past,
            "unique_users_curr": decision.unique_users_curr,
            "label": decision.label,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Never let logging crash the pipeline
            pass


# ----------------------------------------------------------------------------
@dataclass
class _TopKEntry:
    ts: datetime
    items: List[Tuple[int, float]]  # (topic_id, score) sorted with tie policy

    def matured_at(self, delta_hours: int) -> datetime:
        return self.ts + timedelta(hours=delta_hours)


class PrecisionAtKOnline:
    """Online Precision@K with Δ-hour label freeze.

    - Tie-breaking: higher score first, then smaller topic_id.
    - Aggregation window: predictions whose matured time t+Δ is within the
      last `WINDOW_MIN` minutes relative to the latest observed timestamp.
    - Returns a `PrecisionAtKSnapshot` with fields (k5, k10, support).
    """

    def __init__(
        self,
        *,
        delta_hours: Optional[int] = None,
        window_min: Optional[int] = None,
        k_default: Optional[int] = None,
        k_options: Optional[Iterable[int]] = None,
    ) -> None:
        self.delta_hours: int = int(DELTA_HOURS if delta_hours is None else delta_hours)
        self.window_min: int = int(WINDOW_MIN if window_min is None else window_min)
        self.k_default: int = int(K_DEFAULT if k_default is None else k_default)
        self.k_options: Tuple[int, ...] = tuple(k_options or K_OPTIONS)

        self._labels: Dict[int, EmergenceLabelBuffer] = {}
        self._pred_log: Deque[_TopKEntry] = deque()
        self._latest_ts: Optional[datetime] = None

    # ------------------------------------------------------------------
    def record_event(self, *, topic_id: int, user_id: str, ts_iso: str) -> int:
        buf = self._labels.setdefault(topic_id, EmergenceLabelBuffer(
            delta_hours=self.delta_hours, window_min=self.window_min
        ))
        ts = datetime.fromisoformat(ts_iso)
        self._latest_ts = max(self._latest_ts or ts, ts)
        return buf.add_event(ts_iso=ts_iso, user_id=user_id)

    # ------------------------------------------------------------------
    def record_predictions(self, *, ts_iso: str, items: Iterable[Tuple[int, float]]) -> None:
        ts = datetime.fromisoformat(ts_iso)
        self._latest_ts = max(self._latest_ts or ts, ts)
        sorted_items = sorted(items, key=lambda x: (-float(x[1]), int(x[0])))
        self._pred_log.append(_TopKEntry(ts=ts, items=sorted_items))

    # ------------------------------------------------------------------
    def rolling_hourly_scores(self) -> PrecisionAtKSnapshot:
        now = self._latest_ts or datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_min)
        matured: List[_TopKEntry] = []
        for entry in self._pred_log:
            mat = entry.matured_at(self.delta_hours)
            if window_start <= mat <= now:
                matured.append(entry)

        if not matured:
            return {"k5": 0.0, "k10": 0.0, "support": 0}

        k5_sum = 0.0
        k10_sum = 0.0
        for entry in matured:
            k5_sum += self._precision_for_entry(entry, k=5)
            k10_sum += self._precision_for_entry(entry, k=10)

        n = len(matured)
        return {"k5": k5_sum / n, "k10": k10_sum / n, "support": n}

    # ----------------------- Internals --------------------------------
    def _precision_for_entry(self, entry: _TopKEntry, *, k: int) -> float:
        topk = [tid for (tid, _s) in entry.items[:k]]
        if not topk:
            return 0.0
        tp = 0
        for tid in topk:
            buf = self._labels.get(tid)
            if buf is None:
                continue
            if buf.emergent_within(t=entry.ts, delta_hours=self.delta_hours):
                tp += 1
        return tp / float(k)

    # ------------------------------------------------------------------
    def append_snapshot_jsonl(self, *, filename: str = "precision_at_k.jsonl") -> str | None:
        """Compute snapshot and append a JSONL record for dashboards.

        The record schema is minimal and append-only to avoid blocking
        Streamlit consumers:
            {"ts": ISO8601, "precision_at_k": PrecisionAtKSnapshot}

        Args:
            filename: Name of the JSONL file within METRICS_SNAPSHOT_DIR.

        Returns:
            The full path written to on success; None on best-effort failure.
        """
        record = {
            "ts": (self._latest_ts or datetime.utcnow()).isoformat(timespec="seconds"),
            "precision_at_k": self.rolling_hourly_scores(),
        }
        try:
            os.makedirs(METRICS_SNAPSHOT_DIR, exist_ok=True)
            path = os.path.join(METRICS_SNAPSHOT_DIR, filename)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            return path
        except Exception:
            # Never block the pipeline on IO errors
            return None
