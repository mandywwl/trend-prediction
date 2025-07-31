from collectors.twitter import start_twitter_stream, fake_twitter_stream
from collectors.youtube import start_youtube_api_collector
from graph.builder import GraphBuilder
import json
import os
import pickle
import threading

# Directory to save data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# --- Credentials and configuration ---
TWITTER_BEARER_TOKEN = "Your_Twitter_Bearer_Token" # XXX: Replace with your actual Twitter/X Bearer Token (Paid versions only as of 2025)
YOUTUBE_API_KEY = "AIzaSyBCiebLZPuGWg0plQJQ0PP6WbZsv0etacs"  # XXX: Replace with your actual YouTube API Key
KEYWORDS = ["#trending", "fyp", "viral"]  # XXX: Adjust keywords as needed; Applies to Twitter/X stream

# Initialize graph builder
graph = GraphBuilder()
event_counter = 0

# ---- Utility functions ----
def save_graph(graph_obj, filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(graph_obj, f)
    print(f"Graph saved to {path}")

def save_event(event, filename="events.jsonl"):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")

def load_graph(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# ---- Event handler ----
def handle_event(event):
    global event_counter
    graph.process_event(event)
    event_counter += 1

    # Save event for future use (JSON for simplicity now)
    # NOTE: Consider using a database or more structured storage for production
    save_event(event)

    # Save checkpoint every 100 events
    if event_counter % 100 == 0:
        save_graph(graph.G, f"checkpoint_{event_counter}.pk1")
    # print(f"Event from {event['source']}: {event['type']}, updated graph.")

    # TODO: Add optional features: save snapshots, stats, error handling, etc.

# ---- Run collectors ----
def run_twitter():
    # start_twitter_stream(TWITTER_BEARER_TOKEN, KEYWORDS, handle_event) # XXX: Uncomment to use Twitter API
    fake_twitter_stream(keywords=KEYWORDS, on_event=handle_event) # XXX: Simulate Twitter stream for testing (Comment out if using Twitter API)

def run_youtube():
    start_youtube_api_collector(YOUTUBE_API_KEY, on_event=handle_event) # NOTE: Default categories are set in the collector

# ---- Start collectors in separate threads ----
if __name__ == "__main__":
    # Create threads for each collector
    t1 = threading.Thread(target=run_twitter)
    t2 = threading.Thread(target=run_youtube)

    t1.start()
    t2.start()

    # Wait for both to finish
    t1.join()
    t2.join()

    # FOR TESTING: SAVE the graph for manual inspection
    save_graph(graph.G, "test_graph.pkl")
    print("Graph saved at end of script.")
    