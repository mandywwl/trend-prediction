from datetime import datetime, timedelta

from model.inference.spam_filter import SpamScorer
from data_pipeline.storage.builder import GraphBuilder
from service.main import EventHandler
from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder


def _spam_user():
    now = datetime.utcnow()
    return {
        "created_at": (now - timedelta(days=1)).isoformat(),
        "followers": 5,
        "following": 500,
        "posts": [
            {"timestamp": (now - timedelta(minutes=10)).isoformat(), "text": "BUY NOW"},
            {"timestamp": (now - timedelta(minutes=5)).isoformat(), "text": "BUY NOW"},
        ],
    }


def test_spam_scorer_flags_spam_accounts():
    scorer = SpamScorer()
    score = scorer.score_account(_spam_user())
    assert 0.0 <= score <= 1.0
    assert score < 0.3


def test_graph_builder_downweights_spam_edges():
    scorer = SpamScorer()
    user = _spam_user()
    ts = datetime.utcnow()
    builder = GraphBuilder(reference_time=ts, spam_scorer=scorer)
    event = {
        "timestamp": ts.isoformat(),
        "source": "twitter",
        "type": "original",
        "user_id": "u1",
        "content_id": "c1",
        "hashtags": [],
        "user": user,
    }
    builder.process_event(event)
    assert builder.msg[0][0] < 1.0


def test_event_handler_applies_edge_weight():
    scorer = SpamScorer()
    embedder = RealtimeTextEmbedder(batch_size=1, max_latency_ms=5, device="cpu")
    captured = {}

    def fake_infer(ev):
        captured["event"] = ev

    handler = EventHandler(embedder, fake_infer, spam_scorer=scorer)
    handler.handle({"text": "hi", "user": _spam_user()})
    weight = captured["event"]["features"].get("edge_weight")
    assert weight is not None and weight < 1.0
