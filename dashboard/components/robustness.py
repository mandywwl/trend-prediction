from __future__ import annotations

"""
Robustness panel component for spam detection monitoring.

Renders θ_g/θ_u timeline, spam rate, and down-weighted edge percentage.
Use `render_panel()` to get matplotlib figure, metrics, and alert status.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os

import matplotlib.pyplot as plt

from config.config import (
    SPAM_RATE_SPIKE,
    DELTA_HOURS,
    WINDOW_MIN,
    METRICS_SNAPSHOT_DIR,
)


# ---- Data models (lightweight, tolerant of missing fields) ----
@dataclass
class ThetaPoint:
    ts: datetime
    theta_g: float
    theta_u: float


@dataclass
class RobustnessSnapshot:
    ts: datetime
    spam_rate: Optional[float] = None
    downweighted_pct: Optional[float] = None
    theta_g: Optional[float] = None
    theta_u: Optional[float] = None


# ---- File readers ----
def _parse_iso(s: str) -> Optional[datetime]:
    try:
        # datetime.fromisoformat handles 'Z' if replaced; normalize
        s_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_norm)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return []


def _read_adaptive_log(log_path: Path) -> List[RobustnessSnapshot]:
    """Read `datasets/adaptive_thresholds.log` entries into snapshots.

    Expected records include actions like `thresholds_updated` with fields
    `theta_g`, `theta_u`, and optionally `spam_rate`.
    """
    snapshots: List[RobustnessSnapshot] = []
    for rec in _read_jsonl(log_path):
        ts = _parse_iso(str(rec.get("ts", "")))
        if ts is None:
            continue
        snap = RobustnessSnapshot(
            ts=ts,
            spam_rate=float(rec["spam_rate"]) if "spam_rate" in rec else None,
            theta_g=float(rec["theta_g"]) if "theta_g" in rec else None,
            theta_u=float(rec["theta_u"]) if "theta_u" in rec else None,
        )
        snapshots.append(snap)
    return snapshots


def _read_hourly_metrics(dir_path: Path) -> List[RobustnessSnapshot]:
    """Read metrics snapshots if the sink provides robustness fields.

    The schema is flexible: we attempt to read keys from `robustness` nested
    object when present (e.g., spam_rate, downweighted_pct, theta_g, theta_u).
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    snapshots: List[RobustnessSnapshot] = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        # timestamp from file metadata if `meta.generated_at` missing
        ts_iso = (
            (data.get("meta", {}) or {}).get("generated_at")
            or (data.get("meta", {}) or {}).get("ts")
        )
        ts = _parse_iso(ts_iso) if isinstance(ts_iso, str) else None
        if ts is None:
            ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)

        rb = data.get("robustness", {}) or {}
        snapshots.append(
            RobustnessSnapshot(
                ts=ts,
                spam_rate=_to_float_or_none(rb.get("spam_rate")),
                downweighted_pct=_to_float_or_none(rb.get("downweighted_pct")),
                theta_g=_to_float_or_none(rb.get("theta_g")),
                theta_u=_to_float_or_none(rb.get("theta_u")),
            )
        )
    return snapshots


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _read_events(events_path: Path, since: datetime) -> Tuple[int, int]:
    """Return (total_edges, downweighted_edges) for events since `since`.

    An event is considered down-weighted if `features.edge_weight < 1.0`.
    """
    total = 0
    down = 0
    for rec in _read_jsonl(events_path):
        # time field can be `ts_iso` (schema) or `timestamp` (collectors)
        ts_raw = rec.get("ts_iso") or rec.get("timestamp") or rec.get("ts")
        ts = _parse_iso(ts_raw) if isinstance(ts_raw, str) else None
        if ts is None or ts < since:
            continue
        total += 1
        feats = rec.get("features") or {}
        w = feats.get("edge_weight")
        try:
            if w is not None and float(w) < 1.0:
                down += 1
        except Exception:
            pass
    return total, down


# ---- Aggregation utilities ----
def _last_24h_filter(snapshots: List[RobustnessSnapshot]) -> List[RobustnessSnapshot]:
    if not snapshots:
        return []
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=24)
    return [s for s in snapshots if s.ts >= cutoff]


def _theta_series_from_logs(snapshots: List[RobustnessSnapshot]) -> List[ThetaPoint]:
    points: List[ThetaPoint] = []
    for s in snapshots:
        if s.theta_g is None or s.theta_u is None:
            continue
        points.append(ThetaPoint(ts=s.ts, theta_g=float(s.theta_g), theta_u=float(s.theta_u)))
    return points


def _latest_spam_rate(sources: List[List[RobustnessSnapshot]]) -> Optional[float]:
    """Return the most recent spam_rate across provided sources."""
    latest: Tuple[Optional[datetime], Optional[float]] = (None, None)
    for snaps in sources:
        for s in snaps:
            if s.spam_rate is None:
                continue
            if latest[0] is None or s.ts > latest[0]:
                latest = (s.ts, float(s.spam_rate))
    return latest[1]


def _recent_raise_detected(theta_points: List[ThetaPoint]) -> bool:
    """Detect whether θ was raised within the last Δ hours."""
    if len(theta_points) < 2:
        return False
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=DELTA_HOURS)
    # Look for any positive jump in θ_g or θ_u after cutoff
    prev_g = theta_points[0].theta_g
    prev_u = theta_points[0].theta_u
    for p in theta_points:
        if p.ts < cutoff:
            prev_g, prev_u = p.theta_g, p.theta_u
            continue
        if p.theta_g > prev_g + 1e-9 or p.theta_u > prev_u + 1e-9:
            return True
        prev_g, prev_u = p.theta_g, p.theta_u
    return False


# ---- Rendering ----
def render_theta_figure(points: List[ThetaPoint]) -> plt.Figure:
    """Create a Matplotlib figure for θ_g/θ_u timeline (last 24h)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    if not points:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    xs = [p.ts for p in points]
    g = [p.theta_g for p in points]
    u = [p.theta_u for p in points]

    ax.plot(xs, g, label=r"θ_g", color="#1f77b4")
    ax.plot(xs, u, label=r"θ_u", color="#ff7f0e")
    ax.set_title("Adaptive Thresholds (last 24h)")
    ax.set_ylabel("θ value")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    return fig


def render_panel(
    *,
    datasets_dir: str | os.PathLike[str] = "datasets",
    metrics_dir: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Render the robustness panel from logs/metrics.

    Args:
        datasets_dir: Base datasets directory holding logs (default: `datasets`).
        metrics_dir: Optional metrics snapshots dir. Defaults to
            `config.config.METRICS_SNAPSHOT_DIR` when `None`.

    Returns:
        Dict with keys: `figure`, `spam_rate`, `downweighted_pct`, `alert`, `tooltips`.
    """

    base = Path(datasets_dir)
    adaptive_log = base / "adaptive_thresholds.log"
    events_log = base / "events.jsonl"

    # Read sources
    log_snaps = _read_adaptive_log(adaptive_log)
    met_dir = Path(metrics_dir) if metrics_dir else Path(METRICS_SNAPSHOT_DIR)
    metrics_snaps = _read_hourly_metrics(met_dir)

    # Last 24h θ timeline from logs only (higher resolution)
    theta_points = _theta_series_from_logs(_last_24h_filter(log_snaps))
    figure = render_theta_figure(theta_points)

    # Current spam rate (prefer metrics snapshots if they exist and are fresher)
    spam_rate = _latest_spam_rate([
        _last_24h_filter(metrics_snaps),
        _last_24h_filter(log_snaps),
    ])

    # Down-weighted % from metrics; if absent, compute from events.jsonl (last 24h)
    downweighted_pct: Optional[float] = None
    latest_dw = None  # try metrics snapshot first
    latest_ts = None
    for s in _last_24h_filter(metrics_snaps):
        if s.downweighted_pct is None:
            continue
        if latest_ts is None or s.ts > latest_ts:
            latest_ts, latest_dw = s.ts, float(s.downweighted_pct)
    if latest_dw is not None:
        downweighted_pct = latest_dw
    else:
        since = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        tot, down = _read_events(events_log, since)
        if tot > 0:
            downweighted_pct = (down / tot) * 100.0
        else:
            downweighted_pct = 0.0

    # Alerts: spike if spam_rate >= SPAM_RATE_SPIKE or recent θ raise
    raise_detected = _recent_raise_detected(theta_points)
    alert_level = "ok"
    alert_msg = ""
    if spam_rate is not None and spam_rate >= SPAM_RATE_SPIKE:
        alert_level = "alert"
        alert_msg = f"Spam rate {spam_rate:.0%} ≥ spike threshold {SPAM_RATE_SPIKE:.0%}."
    elif raise_detected:
        alert_level = "warn"
        alert_msg = "Thresholds recently raised due to spam pressure."

    tooltips = {
        "spam_rate_spike": SPAM_RATE_SPIKE,
        "delta_hours": DELTA_HOURS,
        "window_min": WINDOW_MIN,
    }

    return {
        "figure": figure,
        "spam_rate": spam_rate,
        "downweighted_pct": downweighted_pct,
        "alert": {"level": alert_level, "message": alert_msg},
        "tooltips": tooltips,
    }

