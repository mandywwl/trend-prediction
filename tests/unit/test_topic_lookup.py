"""Tests for topic lookup persistence."""

import json
from datetime import datetime, timezone

from service.runtime_glue import RuntimeGlue, RuntimeConfig


class DummyHandler:
    def on_event(self, event):
        pass


def test_topic_lookup_persistence(tmp_path):
    lookup_path = tmp_path / "topic_lookup.json"
    metrics_lookup_path = tmp_path / "metrics_lookup.json"
    cache_path = tmp_path / "cache.json"
    metrics_dir = tmp_path / "metrics"

    config = RuntimeConfig(
        predictions_cache_path=str(cache_path),
        metrics_snapshot_dir=str(metrics_dir),
        topic_lookup_path=str(lookup_path),
        metrics_lookup_path=str(metrics_lookup_path),
    )
    glue = RuntimeGlue(event_handler=DummyHandler(), config=config)

    event = {
        "event_id": "123",
        "actor_id": "u1",
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }
    glue._record_event_for_metrics(event, {"topicA": 0.9, "456": 0.5})

    with open(lookup_path, "r", encoding="utf-8") as f:
        topic_mapping = json.load(f)

    with open(metrics_lookup_path, "r", encoding="utf-8") as f:
        metrics_mapping = json.load(f)

    # topic_lookup should only contain human-readable labels
    assert str(hash("123") % 1_000_000) not in topic_mapping
    assert str(abs(hash("456")) % 1_000_000) not in topic_mapping
    assert topic_mapping[str(abs(hash("topicA")) % 1_000_000)] == "topicA"

    # metrics_lookup should contain all labels
    assert metrics_mapping[str(hash("123") % 1_000_000)] == "123"
    assert metrics_mapping[str(abs(hash("topicA")) % 1_000_000)] == "topicA"
    assert metrics_mapping[str(abs(hash("456")) % 1_000_000)] == "456"

