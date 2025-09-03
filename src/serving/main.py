""" Main entry point for the serving layer."""

from collectors.twitter import fake_twitter_stream
from collectors.youtube import start_youtube_api_collector
from collectors.google_trends import start_google_trends_collector
from features.text_rt_distilbert import RealtimeTextEmbedder
from graph.builder import GraphBuilder
from robustness.spam_filter import SpamScorer
from robustness.adaptive_thresholds import SensitivityController
from serving.event_handler import EventHandler
from utils.io import ensure_dir
from pathlib import Path
import json
import os
import threading
import torch

try:
    from data.preprocessing import build_tgn
except Exception:
    build_tgn = None


# Directory to save data
DATA_DIR = ensure_dir(Path(__file__).resolve().parents[2] / "datasets")

# --- Credentials and configuration ---
# TODO (for production): Shift to environment variables or secure vaults
TWITTER_BEARER_TOKEN = "Your_Twitter_Bearer_Token"  # XXX: Replace with your actual Twitter/X Bearer Token (Paid versions only as of 2025)
YOUTUBE_API_KEY = "AIzaSyBCiebLZPuGWg0plQJQ0PP6WbZsv0etacs"  # XXX: Replace with your actual YouTube API Key # TODO: Remove before submission
KEYWORDS = [
    "#trending",
    "fyp",
    "viral",
]  # XXX: Adjust keywords as needed; Applies to Twitter/X stream

# Initialise components
spam_scorer = SpamScorer()
graph = GraphBuilder(spam_scorer=spam_scorer)
sensitivity = SensitivityController()
embedder = RealtimeTextEmbedder(batch_size=8, max_latency_ms=50, device="cpu")

# Wire runtime handler (preprocess + adaptive back-pressure)
event_counter = 0


def _infer_into_graph(event):
    """Minimal inference hook: push event into graph builder.

    This can later be swapped for a TGN inference call. Keeping a separate
    function allows EventHandler to remain decoupled from graph internals.
    """
    graph.process_event(event)


handler = EventHandler(
    embedder,
    _infer_into_graph,
    spam_scorer=spam_scorer,
    sensitivity=sensitivity,
)


# ---- Utility functions ----
def save_graph(graph_obj, filename):
    path = DATA_DIR / filename
    torch.save(graph_obj, path)
    print(f"Graph saved to {path}")


def save_event(event, filename="events.jsonl"):
    path = DATA_DIR / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def load_graph(filename):
    path = DATA_DIR / filename
    return torch.load(path)


# ---- Event handler ----
def handle_event(event):
    """Route incoming event through adaptive handler and persist artifacts."""
    global event_counter
    handler.handle(event)
    event_counter += 1

    # Save raw event (append-only) for dashboards and reproducibility
    # TODO (for production): Consider using database or more structured storage
    save_event(event)
    save_event(event)

    # Save a lightweight checkpoint periodically (TemporalData)
    if event_counter % 100 == 0:
        save_graph(graph.to_temporal_data(), f"checkpoint_{event_counter}.pt")
    # TODO (for production): Add optional stats/error handling, hook controller metrics to dashboards


# ---- Run collectors ----
def run_twitter():
    # start_twitter_stream(TWITTER_BEARER_TOKEN, KEYWORDS, handle_event) # XXX: Uncomment to use Twitter API
    fake_twitter_stream(
        keywords=KEYWORDS, on_event=handle_event
    )  # XXX: Simulate Twitter stream for testing (Comment out if using Twitter API)


def run_youtube():
    start_youtube_api_collector(
        YOUTUBE_API_KEY, on_event=handle_event
    )  # NOTE: Default categories are set in the collector


def run_googletrends():
    start_google_trends_collector(on_event=handle_event)


# ---- Start collectors in separate threads ----
if __name__ == "__main__":
    # Ensure preprocessing output exists (auto-run if missing). Set PREPROCESS_FORCE=1 to force rebuild.
    tgn_file = DATA_DIR / "tgn_edges_basic.npz"
    try:
        if build_tgn:
            if (not tgn_file.exists()) or os.environ.get("PREPROCESS_FORCE") == "1":
                print("[main] Running preprocessing (build_tgn)...")
                build_tgn()
            else:
                print(
                    "[main] TGN file exists, skipping preprocessing. Set PREPROCESS_FORCE=1 to force."
                )
        else:
            print("[main] build_tgn not available; skipping preprocessing.")
    except Exception as e:
        print(f"[main] Preprocessing failed but continuing: {e}")

    # Create threads for each collector
    t1 = threading.Thread(target=run_twitter)
    t2 = threading.Thread(target=run_youtube)
    t3 = threading.Thread(target=run_googletrends)

    t1.start()
    t2.start()
    t3.start()

    # Wait for both to finish
    t1.join()
    t2.join()
    t3.join()

    # FOR TESTING: SAVE the graph for manual inspection
    save_graph(graph.to_temporal_data(), "test_graph.pt")
    print("Graph saved at end of script.")
