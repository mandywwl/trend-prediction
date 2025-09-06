import pytest

from model.inference.drift_monitor import PerformanceDriftMonitor


def test_drift_monitor_alerts_on_drop():
    alerts: list[str] = []
    monitor = PerformanceDriftMonitor(
        window=2,
        f1_threshold=0.75,
        precision_k_threshold=0.5,
        k=2,
        alert_fn=alerts.append,
    )

    # Initial high-performance update
    monitor.update(
        y_true=[1, 0, 1],
        y_pred=[1, 0, 1],
        topk_pred=[1, 2],
        relevant_items=[1],
    )
    assert alerts == []  # no alert yet

    # Second update with poor performance should trigger alerts
    monitor.update(
        y_true=[1, 1, 1],
        y_pred=[0, 0, 0],
        topk_pred=[3, 4],
        relevant_items=[5],
    )
    assert any("F1" in msg for msg in alerts)
    assert any("Precision@2" in msg for msg in alerts)
