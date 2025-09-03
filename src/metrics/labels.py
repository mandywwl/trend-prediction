from __future__ import annotations

"""Trend emergence labelling with adaptive thresholds.

Reads DELTA_HOURS and WINDOW_MIN from config. Fetches (theta_g, theta_u)
from SensitivityController, applies them to scale baseline thresholds for
growth and unique users, and logs applied values per decision for replay.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterable, Iterator, Optional, Tuple
from collections import deque
import json
import os

from config.config import DELTA_HOURS, WINDOW_MIN
from robustness.adaptive_thresholds import SensitivityController


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
        sensitivity: SensitivityController,
        *,
        growth_factor_base: float = 2.0,
        unique_users_base: int = 50,
        delta_hours: Optional[int] = None,
        window_min: Optional[int] = None,
        log_path: str = os.path.join("data", "emergence_labels.log"),
    ) -> None:
        self.sensitivity = sensitivity
        self.growth_factor_base = float(growth_factor_base)
        self.unique_users_base = int(unique_users_base)
        self.delta_hours = int(DELTA_HOURS if delta_hours is None else delta_hours)
        self.window_min = int(WINDOW_MIN if window_min is None else window_min)
        self.log_path = log_path

        self._events: Deque[Tuple[datetime, str]] = deque()
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

        self._log(EmergenceDecision(
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
        ))

        return label

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

                # Recompute label using the same thresholds and constants
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

