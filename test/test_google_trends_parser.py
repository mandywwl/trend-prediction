import sys
from pathlib import Path

# Ensure src is on the path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.event_parser import parse_event


def test_google_trends_context_parsing():
    event = {
        "timestamp": "2025-01-01T00:00:00",
        "source": "google_trends",
        "type": "trend",
        "content_id": "trend_sample_term",
        "context": ["Breaking News", "Some Article"],
    }
    edges = parse_event(event)
    assert ("trend-event", None, "trend_sample_term", "trend") in edges
    context_edges = [e for e in edges if e[0] == "trend-context"]
    assert len(context_edges) == 2
    assert all(e[1] == "trend_sample_term" for e in context_edges)
    assert {e[2] for e in context_edges} == {"ctx_breaking_news", "ctx_some_article"}
