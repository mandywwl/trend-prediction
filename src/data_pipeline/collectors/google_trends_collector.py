from datetime import datetime, timezone
import time
# Try to import the pytrends library
try:
    from pytrends.request import TrendReq
except Exception:  # ImportError or other errors
    TrendReq = None

from utils.text import slugify as _slugify


def start_google_trends_collector(
    on_event=None,
    region: str = "US",
    category: str = "all",
    count: int = 20,
    interval: int = 3600,
):
    """Poll Google Trends' realtime trending searches every ``interval`` seconds.

    Parameters
    ----------
    on_event: callable
        Callback to handle each emitted event.
    region: str
        Two-letter region code, e.g. ``"US"``.
    category: str
        Trend category. ``"all"`` to get all categories.
    count: int
        Number of results to process from the API.
    interval: int
        Seconds between polls. Defaults to one hour.
    """
    if TrendReq is None:
        raise ImportError("pytrends is required for the Google Trends collector")

    print("[Google Trends Collector] Starting stream...")
    pytrends = TrendReq(hl="en-US", tz=360)

    while True:
        try:
            df = pytrends.realtime_trending_searches(
                pn=region, cat=category, count=count
            )
            for _, row in df.iterrows():
                data = row.to_dict()
                term = data.get("query") or data.get("title")
                if not term:
                    continue
                contexts = []
                entity_names = data.get("entityNames")
                if isinstance(entity_names, list):
                    contexts.extend([str(c) for c in entity_names])
                articles = data.get("articles")
                if isinstance(articles, list):
                    for art in articles:
                        title = art.get("title") if isinstance(art, dict) else str(art)
                        if title:
                            contexts.append(title)
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content_id": f"trend_{_slugify(term)}",
                    "source": "google_trends",
                    "type": "trend",
                    "text": term,
                    "context": contexts,
                }
                if on_event:
                    on_event(event)
                else:
                    print(event)
        except Exception as e:
            print(f"[Google Trends Collector] Error fetching trends: {e}")
        time.sleep(interval)


def fake_google_trends_stream(on_event=None, n_events: int = 5, delay: float = 1.0):
    """Emit synthetic Google Trends events for offline testing."""
    print("[Fake Google Trends Stream] Starting simulation...")
    for i in range(n_events):
        term = f"Example Trend {i}"
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_id": f"trend_{_slugify(term)}",
            "source": "google_trends",
            "type": "trend",
            "text": term,
            "context": [f"Context {i}", "General News"],
        }
        if on_event:
            on_event(event)
        else:
            print(event)
        time.sleep(delay)
    print("[Fake Google Trends Stream] Finished emitting events.")
