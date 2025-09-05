"""
Lightweight end-to-end ingestion simulation using the existing service components.

Runs the fake Twitter stream and a short fake Google Trends burst, routes events
through the EventHandler/GraphBuilder, then saves a TemporalData checkpoint.
"""

from pathlib import Path
import time

from data_pipeline.collectors.twitter_collector import fake_twitter_stream
from data_pipeline.collectors.google_trends_collector import fake_google_trends_stream
from data_pipeline.storage.builder import GraphBuilder
from model.inference.spam_filter import SpamScorer
from model.inference.adaptive_thresholds import SensitivityController
from data_pipeline.processors.text_rt_distilbert import RealtimeTextEmbedder
from service.services.preprocessing.event_handler import EventHandler
from utils.io import ensure_dir


def main():
    data_dir = ensure_dir(Path(__file__).resolve().parents[1] / "datasets")

    spam_scorer = SpamScorer()
    graph = GraphBuilder(spam_scorer=spam_scorer)
    sensitivity = SensitivityController()
    embedder = RealtimeTextEmbedder(batch_size=4, max_latency_ms=25, device="cpu")

    def _infer_into_graph(event):
        graph.process_event(event)

    handler = EventHandler(
        embedder,
        _infer_into_graph,
        spam_scorer=spam_scorer,
        sensitivity=sensitivity,
    )

    def on_event(event):
        handler.handle(event)

    # Simulate a small burst from Twitter (10 events) and Google Trends (5 events)
    fake_twitter_stream(on_event=on_event, n_events=10, delay=0.05)
    fake_google_trends_stream(on_event=on_event, n_events=5, delay=0.05)

    # Save graph snapshot
    out = data_dir / "test_graph_sim.pt"
    import torch

    torch.save(graph.to_temporal_data(), out)
    print(f"Saved simulated TemporalData to {out}")


if __name__ == "__main__":
    main()

