import numpy as np

from embeddings.rt_distilbert import RealtimeTextEmbedder
from runtime.event_handler import EmbeddingPreprocessor, EventHandler


def test_preprocessor_adds_embedding():
    embedder = RealtimeTextEmbedder(batch_size=1, max_latency_ms=5, device="cpu")
    pre = EmbeddingPreprocessor(embedder)
    event = {"text": "hello world"}
    out = pre(event)
    assert "text_emb" in out.setdefault("features", {})
    assert len(out["features"]["text_emb"]) == embedder.model.config.hidden_size


def test_event_handler_runs_preprocessor_before_infer():
    embedder = RealtimeTextEmbedder(batch_size=1, max_latency_ms=5, device="cpu")
    captured = {}

    def fake_infer(ev):
        captured["event"] = ev

    handler = EventHandler(embedder, fake_infer)
    handler.handle({"caption": "something"})
    assert "event" in captured
    assert "text_emb" in captured["event"]["features"]
