"""Adaptive sensitivity and back-pressure controller."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Deque, Dict, Iterable, Optional
from collections import deque
import json
from pathlib import Path

from utils.path_utils import find_repo_root

from config.config import (
    SPAM_RATE_SPIKE,
    THRESH_RAISE_FACTOR,
    THRESH_DECAY_RATE,
    SPAM_WINDOW_MIN,
    SLO_P95_MS,
    SLO_MED_MS,
)


@dataclass
class Thresholds:
    """Current adaptive thresholds."""

    theta_g: float
    theta_u: float


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity and adaptation behaviour.

    Attributes:
        baseline_theta_g: Baseline global threshold theta_g.
        baseline_theta_u: Baseline user threshold theta_u.
        window_size: Sliding window size (number of recent events) used for
            spam-rate and latency statistics. Keep small to react quickly.
        raise_factor: Multiplier applied to thresholds when spam_rate > 0.2.
            The raise is applied to the current values, capped by
            max_multiplier_of_baseline.
        max_multiplier_of_baseline: Absolute cap relative to baselines to
            prevent runaway growth.
        decay_alpha: Exponential decay factor towards baseline when spam
            pressure subsides. 0 < decay_alpha < 1; higher means faster decay.
        log_path: Filepath for append-only JSONL logs of changes.
    """

    baseline_theta_g: float = 0.5
    baseline_theta_u: float = 0.5
    window_size: int = 200
    raise_factor: float = THRESH_RAISE_FACTOR
    max_multiplier_of_baseline: float = 2.0
    decay_alpha: float = THRESH_DECAY_RATE
    log_path: Path = find_repo_root() / "datasets" / "adaptive_thresholds.log"
    window_minutes: int = SPAM_WINDOW_MIN
    log_snapshot_every_events: int = 50        # emit after N events (even if unchanged)
    log_snapshot_every_secs: int = 120         # or after N seconds since last snapshot


@dataclass
class SLOs:
    """Latency service level objectives in milliseconds."""

    p50_ms: int = SLO_MED_MS
    p95_ms: int = SLO_P95_MS


@dataclass
class BackPressurePolicy:
    """Current back-pressure policy recommendations.

    Attributes:
        active: When True, latency SLOs are currently breached.
        heavy_ops_enabled: If False, callers should skip expensive operations
            (e.g., long-seq embeddings, graph augmentations) until recovered.
        sampler_size: Recommended sampler size (e.g., subsampling rate or
            batch size for upstream components). Callers decide how to map
            this recommendation to their own knobs.
    """

    active: bool
    heavy_ops_enabled: bool
    sampler_size: int


class SensitivityController:
    """Adaptive controller for thresholds and back-pressure.

    Usage:
        ctrl = SensitivityController()
        ctrl.record_event(is_spam=True, latency_ms=1200)
        th = ctrl.thresholds()  # (theta_g, theta_u)
        policy = ctrl.policy()  # back-pressure policy
        stats = ctrl.metrics()  # spam_rate, p50, p95
    """

    def __init__(
        self,
        *,
        config: Optional[SensitivityConfig] = None,
        slos: Optional[SLOs] = None,
        initial_sampler_size: int = 8,
        min_sampler_size: int = 2,
        max_sampler_size: int = 64,
    ) -> None:
        self.cfg = config or SensitivityConfig()
        self.cfg.log_path = Path(self.cfg.log_path)
        self.slos = slos or SLOs()

        # Sliding windows
        self._spam_window: Deque[int] = deque(maxlen=self.cfg.window_size)
        self._latency_window: Deque[float] = deque(maxlen=self.cfg.window_size)

        # Adaptive thresholds
        self._theta_g: float = self.cfg.baseline_theta_g
        self._theta_u: float = self.cfg.baseline_theta_u

        # Back-pressure state
        self._policy = BackPressurePolicy(
            active=False,
            heavy_ops_enabled=True,
            sampler_size=initial_sampler_size,
        )
        self._min_sampler = min_sampler_size
        self._max_sampler = max_sampler_size
        self._cooldown_until: Optional[datetime] = None

        # Logging
        self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)

        # snapshot cadence state
        self._events_since_snapshot: int = 0
        self._last_snapshot_ts: Optional[datetime] = None

        self._lock = RLock()

        self._log_thresholds_snapshot(reason="bootstrap") # write one baseline snapshot immediately so the panel is never empty

    # ------------------------------------------------------------------
    def record_event(self, *, is_spam: bool, latency_ms: Optional[float]) -> None:
        """Record a single event outcome and update controller state.

        Args:
            is_spam: Ground-truth spam label for the event. Do not pre-filter
                or override this value; the controller relies on verbatim truth.
            latency_ms: Observed end-to-end latency for the event in
                milliseconds. When None, the event is ignored for latency
                statistics but still influences spam-rate.
        """
        with self._lock:
            self._spam_window.append(1 if is_spam else 0)
            if latency_ms is not None:
                self._latency_window.append(float(latency_ms))

            prev_th = (self._theta_g, self._theta_u)
            prev_policy = self._policy

            # Update thresholds based on spam rate
            self._maybe_adjust_thresholds()

            # Update back-pressure policy based on latency percentiles
            self._maybe_apply_back_pressure()

            # # Log changes if any
            # if (self._theta_g, self._theta_u) != prev_th:
            #     self._log(
            #         action="thresholds_updated",
            #         payload={
            #             "theta_g": self._theta_g,
            #             "theta_u": self._theta_u,
            #             "spam_rate": self._spam_rate(),
            #         },
            #     )

            # Log on change or cadence
            changed = (self._theta_g, self._theta_u) != prev_th
            self._events_since_snapshot += 1
            now = datetime.now(timezone.utc)
            due_by_events = self._events_since_snapshot >= self.cfg.log_snapshot_every_events
            due_by_time = (
                self._last_snapshot_ts is None
                or (now - self._last_snapshot_ts).total_seconds() >= self.cfg.log_snapshot_every_secs
            )
            if changed or due_by_events or due_by_time:
                self._log_thresholds_snapshot(reason="changed" if changed else "periodic")

            if (
                (self._policy.active != prev_policy.active)
                or (self._policy.heavy_ops_enabled != prev_policy.heavy_ops_enabled)
                or (self._policy.sampler_size != prev_policy.sampler_size)
            ):
                self._log(
                    action="policy_updated",
                    payload={
                        "active": self._policy.active,
                        "heavy_ops_enabled": self._policy.heavy_ops_enabled,
                        "sampler_size": self._policy.sampler_size,
                        "p50_ms": self._p50_ms(),
                        "p95_ms": self._p95_ms(),
                    },
                )

    # ------------------------------------------------------------------
    def thresholds(self) -> Thresholds:
        """Return current adaptive thresholds."""
        with self._lock:
            return Thresholds(theta_g=self._theta_g, theta_u=self._theta_u)

    # ------------------------------------------------------------------
    def policy(self) -> BackPressurePolicy:
        """Return current back-pressure policy recommendations."""
        with self._lock:
            # Return a value object to avoid exposing internal references
            return BackPressurePolicy(
                active=self._policy.active,
                heavy_ops_enabled=self._policy.heavy_ops_enabled,
                sampler_size=self._policy.sampler_size,
            )

    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, float]:
        """Return recent window metrics: spam_rate, p50_ms, p95_ms."""
        with self._lock:
            return {
                "spam_rate": self._spam_rate(),
                "p50_ms": self._p50_ms(),
                "p95_ms": self._p95_ms(),
            }
    
    # ------------------------------------------------------------------
    def _log_thresholds_snapshot(self, *, reason: str = "periodic") -> None:
        rec = {
            "theta_g": round(float(self._theta_g), 3),
            "theta_u": round(float(self._theta_u), 3),
            "spam_rate": round(float(self._spam_rate()), 3),
            # the panel only requires ts/theta_g/theta_u[/spam_rate]; extra fields are fine
            "reason": reason,
        }
        self._log(action="thresholds_updated", payload=rec)
        self._events_since_snapshot = 0
        self._last_snapshot_ts = datetime.now(timezone.utc)

    # ----------------------- Internals --------------------------------
    def _spam_rate(self) -> float:
        if not self._spam_window:
            return 0.0
        return sum(self._spam_window) / len(self._spam_window)

    def _percentile(self, data: Iterable[float], pct: float) -> float:
        seq = list(data)
        if not seq:
            return 0.0
        seq.sort()
        k = max(0, min(len(seq) - 1, int(round((pct / 100.0) * (len(seq) - 1)))))
        return float(seq[k])

    def _p50_ms(self) -> float:
        return self._percentile(self._latency_window, 50.0)

    def _p95_ms(self) -> float:
        return self._percentile(self._latency_window, 95.0)

    def _maybe_adjust_thresholds(self) -> None:
        spam_rate = self._spam_rate()
        cap_g = self.cfg.baseline_theta_g * self.cfg.max_multiplier_of_baseline
        cap_u = self.cfg.baseline_theta_u * self.cfg.max_multiplier_of_baseline

        if spam_rate > SPAM_RATE_SPIKE:
            # Raise by configured factor relative to current, with cap
            new_g = min(self._theta_g * self.cfg.raise_factor, cap_g)
            new_u = min(self._theta_u * self.cfg.raise_factor, cap_u)
            self._theta_g, self._theta_u = new_g, new_u
        else:
            # Exponential decay back to baseline
            self._theta_g = self._decay_towards(
                self._theta_g, self.cfg.baseline_theta_g, self.cfg.decay_alpha
            )
            self._theta_u = self._decay_towards(
                self._theta_u, self.cfg.baseline_theta_u, self.cfg.decay_alpha
            )

    def _decay_towards(self, value: float, target: float, alpha: float) -> float:
        if value == target:
            return value
        # Move a fraction alpha of the gap back towards the target
        return target + (value - target) * (1.0 - alpha)

    def _maybe_apply_back_pressure(self) -> None:
        now = datetime.now(timezone.utc)
        p95 = self._p95_ms()

        if p95 > self.slos.p95_ms:
            # Enter or reinforce back-pressure
            new_sampler = max(
                self._min_sampler,
                max(self._min_sampler, int(self._policy.sampler_size * 0.7)),
            )
            if (
                new_sampler == self._policy.sampler_size
                and new_sampler > self._min_sampler
            ):
                new_sampler -= 1
            self._policy = BackPressurePolicy(
                active=True, heavy_ops_enabled=False, sampler_size=new_sampler
            )
            # Set a short cooldown so we don't flap (e.g., 2 windows)
            self._cooldown_until = now + self._cooldown_duration()
            return

        # If under SLO and past cooldown, gradually restore
        if (
            self._policy.active
            and (self._cooldown_until is not None)
            and (now >= self._cooldown_until)
        ):
            new_sampler = min(
                self._max_sampler,
                max(
                    self._policy.sampler_size + 1, int(self._policy.sampler_size * 1.2)
                ),
            )
            # If sampler recovered sufficiently and p50 is healthy, re-enable heavy ops
            p50 = self._p50_ms()
            heavy_ok = p50 <= self.slos.p50_ms and p95 <= self.slos.p95_ms
            self._policy = BackPressurePolicy(
                active=False,
                heavy_ops_enabled=heavy_ok,
                sampler_size=new_sampler,
            )
            self._cooldown_until = None

        # If not active and not in cooldown, ensure policy is sane
        if not self._policy.active and self._policy.sampler_size < self._max_sampler:
            # Gentle drift towards default when stable
            self._policy = BackPressurePolicy(
                active=False,
                heavy_ops_enabled=True,
                sampler_size=min(self._max_sampler, self._policy.sampler_size + 1),
            )

    def _cooldown_duration(self) -> timedelta:
        # Heuristic: two window-equivalents worth of observations, assuming
        # events are evenly spaced. Without a clock rate, use a fixed short duration.
        return timedelta(seconds=5)

    def _log(self, *, action: str, payload: Dict[str, object]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "action": action,
            **payload,
        }
        try:
            with open(self.cfg.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Best-effort logging; never raise from controller
            pass

