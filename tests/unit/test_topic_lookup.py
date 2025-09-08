"""Tests for topic lookup persistence."""

import json
from datetime import datetime, timezone

from service.runtime_glue import RuntimeGlue, RuntimeConfig


class DummyHandler:
    def on_event(self, event):
        pass


def test_topic_lookup_persistence(tmp_path):
    lookup_path = tmp_path / "lookup.json"
    cache_path = tmp_path / "cache.json"
    metrics_dir = tmp_path / "metrics"

    config = RuntimeConfig(
        predictions_cache_path=str(cache_path),
        metrics_snapshot_dir=str(metrics_dir),
        topic_lookup_path=str(lookup_path),
    )
    glue = RuntimeGlue(event_handler=DummyHandler(), config=config)

    event = {
        "event_id": "evt1",
        "actor_id": "u1",
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }
    glue._record_event_for_metrics(event, {"topicA": 0.9})

    with open(lookup_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert mapping[str(hash("evt1") % 1_000_000)] == "evt1"
    assert mapping[str(abs(hash("topicA")) % 1_000_000)] == "topicA"

