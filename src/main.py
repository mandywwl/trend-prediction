from collectors.twitter import start_twitter_stream, fake_twitter_stream
from collectors.youtube import start_youtube_api_collector
from collectors.google_trends import start_google_trends_collector
from graph.builder import GraphBuilder
import json
import os
import threading
import torch
try:
    from preprocessing.preprocess_tgn import build_tgn
except Exception:
    try:
        from src.preprocessing.preprocess_tgn import build_tgn
    except Exception:
        build_tgn = None

# Directory to save data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# --- Credentials and configuration ---
# TODO: Shift to environment variables or secure vaults in production
TWITTER_BEARER_TOKEN = "Your_Twitter_Bearer_Token" # XXX: Replace with your actual Twitter/X Bearer Token (Paid versions only as of 2025)
YOUTUBE_API_KEY = "AIzaSyBCiebLZPuGWg0plQJQ0PP6WbZsv0etacs"  # XXX: Replace with your actual YouTube API Key # TODO: Remove before submission
KEYWORDS = ["#trending", "fyp", "viral"]  # XXX: Adjust keywords as needed; Applies to Twitter/X stream

# Initialize graph builder
graph = GraphBuilder()
event_counter = 0

# ---- Utility functions ----
def save_graph(graph_obj, filename):
    path = os.path.join(DATA_DIR, filename)
    torch.save(graph_obj, path)
    print(f"Graph saved to {path}")

def save_event(event, filename="events.jsonl"):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")

def load_graph(filename):
    path = os.path.join(DATA_DIR, filename)
    return torch.load(path)

# ---- Event handler ----
def handle_event(event):
    global event_counter
    graph.process_event(event)
    event_counter += 1

    # Save event for future use (JSON for simplicity now)
    # TODO: Consider using database or more structured storage for production
    save_event(event)

    # Save checkpoint every 100 events
    if event_counter % 100 == 0:
        save_graph(graph.G, f"checkpoint_{event_counter}.pkl")
    # print(f"Event from {event['source']}: {event['type']}, updated graph.")

    # TODO: Add optional features: save snapshots, stats, error handling, etc.

# ---- Run collectors ----
def run_twitter():
    # start_twitter_stream(TWITTER_BEARER_TOKEN, KEYWORDS, handle_event) # XXX: Uncomment to use Twitter API
    fake_twitter_stream(keywords=KEYWORDS, on_event=handle_event) # XXX: Simulate Twitter stream for testing (Comment out if using Twitter API)

def run_youtube():
    start_youtube_api_collector(YOUTUBE_API_KEY, on_event=handle_event) # NOTE: Default categories are set in the collector

def run_googletrends():
    start_google_trends_collector(on_event=handle_event)



# ---- Start collectors in separate threads ----
if __name__ == "__main__":
    # Ensure preprocessing output exists (auto-run if missing). Set PREPROCESS_FORCE=1 to force rebuild.
    tgn_file = os.path.join(DATA_DIR, "tgn_edges_basic.npz")
    try:
        if build_tgn:
            if (not os.path.exists(tgn_file)) or os.environ.get("PREPROCESS_FORCE") == "1":
                print("[main] Running preprocessing (build_tgn)...")
                build_tgn()
            else:
                print("[main] TGN file exists, skipping preprocessing. Set PREPROCESS_FORCE=1 to force.")
        else:
            print("[main] build_tgn not available; skipping preprocessing.")
    except Exception as e:
        print(f"[main] Preprocessing failed but continuing: {e}")

    # Create threads for each collector
    t1 = threading.Thread(target=run_twitter)
    t2 = threading.Thread(target=run_youtube)

    t1.start()
    t2.start()

    # Wait for both to finish
    t1.join()
    t2.join()

    # FOR TESTING: SAVE the graph for manual inspection
    save_graph(graph.to_temporal_data(), "test_graph.pt")
    print("Graph saved at end of script.")