from collectors.twitter_collector import start_twitter_stream, fake_twitter_stream
from collectors.youtube_collector import start_youtube_scraper
from graph.graph_builder import GraphBuilder
import json
import os
import pickle

# ---- Path setup ----
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True) 

# ---- Graph builder and event counter ----
graph = GraphBuilder()
event_counter = 0

TWITTER_BEARER_TOKEN = "Your_Twitter_Bearer_Token" # XXX:Replace with your actual Twitter/X Bearer Token (Paid versions only as of 2025)
KEYWORDS = ["#trending", "fyp", "viral"]  # XXX: Adjust keywords as needed

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
        save_graph(graph.G, f"checkpoint_{event_counter}.gpickle")
    # print(f"Event from {event['source']}: {event['type']}, updated graph.")

    # TODO: Add optional features: save snapshots, stats, error handling, etc.

def main():

    # Start collectors (could be threaded/async for real concurrency)
    # start_twitter_stream(TWITTER_BEARER_TOKEN, KEYWORDS, handle_event) # XXX: Uncomment to use Twitter API
    fake_twitter_stream(keywords=KEYWORDS, on_event=handle_event) # XXX: Simulate Twitter stream for testing (Comment out if using Twitter API)
    start_youtube_scraper(on_event=handle_event)

    # FOR TESTING: SAVE the graph for manual inspection
    save_graph(graph.G, "test_graph.pkl")
    print("Graph saved at end of script.")

if __name__ == "__main__":
    main()