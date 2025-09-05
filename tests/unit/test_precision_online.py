from datetime import datetime, timedelta

from model.evaluation.metrics import PrecisionAtKOnline


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def test_online_precision_at_k_with_delta_freeze_and_ties():
    base = datetime(2025, 1, 1, 0, 0, 0)
    evalr = PrecisionAtKOnline()

    # t0 predictions with tie scores; smaller topic_id should rank higher
    t0 = base
    preds_t0 = [
        (10, 0.9),
        (2, 0.9),  # same score as topic 10; smaller id -> should rank above 10
        (1, 0.8),
        (3, 0.7),
        (4, 0.6),
        (5, 0.5),
        (6, 0.4),
        (7, 0.3),
        (8, 0.2),
        (9, 0.1),
    ]
    evalr.record_predictions(ts_iso=iso(t0), items=preds_t0)

    # Make topic 2 emergent within (t0, t0+Δ]: add many unique users in current window
    # Create 60 unique users in [t0+30m, t0+60m]; past window [t0-90m, t0-30m] ~ empty
    for i in range(60):
        ts = t0 + timedelta(minutes=30 + i % 30)  # within [30,60]m
        evalr.record_event(topic_id=2, user_id=f"u{i}", ts_iso=iso(ts))

    # Topic 1: not emergent (too few users)
    for i in range(10):
        ts = t0 + timedelta(minutes=45 + i % 10)
        evalr.record_event(topic_id=1, user_id=f"a{i}", ts_iso=iso(ts))

    # Topic 4: emergent but only after Δ (beyond t0+Δ), should not count for t0
    for i in range(55):
        ts = t0 + timedelta(hours=2, minutes=10 + i % 30)
        evalr.record_event(topic_id=4, user_id=f"z{i}", ts_iso=iso(ts))

    # Second prediction at t1 with different ordering
    t1 = base + timedelta(minutes=30)
    preds_t1 = [
        (4, 0.95),  # will become emergent within (t1, t1+Δ]
        (2, 0.80),  # also emergent
        (11, 0.70),
        (12, 0.60),
        (13, 0.50),
        (14, 0.40),
        (15, 0.30),
        (16, 0.20),
        (17, 0.10),
        (18, 0.05),
    ]
    evalr.record_predictions(ts_iso=iso(t1), items=preds_t1)

    # Advance "now" by recording a benign event far in the future to set latest_ts
    now = base + timedelta(hours=3)
    evalr.record_event(topic_id=99, user_id="noop", ts_iso=iso(now))

    snap = evalr.rolling_hourly_scores()
    # Both t0 and t1 matured times fall within (now-60m, now]; support==2
    assert snap["support"] == 2

    # For t0@K=5: top-5 should include topic 2 due to tie-breaking (2 before 10)
    # so P@5 >= 0.2. For t1@K=5: topics 4 and 2 both emergent => 2/5 = 0.4.
    # Average across two matured entries gives non-trivial values.
    assert 0.15 <= snap["k5"] <= 0.5
    assert 0.1 <= snap["k10"] <= 0.5
