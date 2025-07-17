from collectors.twitter_collector import start_twitter_stream
from collectors.twitter_collector import fake_twitter_stream
# from collectors.youtube_collector import start_youtube_scraper
from graph_builder import GraphBuilder
import json
import networkx as nx
from networkx.readwrite import gpickle

graph = GraphBuilder()
event_counter = 0

TWITTER_BEARER_TOKEN = "Your_Twitter_Bearer_Token" # XXX:Replace with your actual Twitter/X Bearer Token (Paid versions only as of 2025)
KEYWORDS = ["#trending", "fyp", "viral"]  # XXX: Adjust keywords as needed


# For each platform, define a callback to process events
def handle_event(event):
    global event_counter
    graph.process_event(event)
    event_counter += 1
    # Save every 100 events
    if event_counter % 100 == 0:
        gpickle.write_gpickle(graph.G, f"../data/checkpoint_{event_counter}.gpickle")
    print(f"Event from {event['source']}: {event['type']}, updated graph.")

    # Save each event for training/future use (JSON for simplicity now)
    # In production, consider using a database or more structured storage
    with open("data/events.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")

    
    # TODO: Add optional features: save snapshots, stats, error handling, etc.

def main():

    # Start collectors (could be threaded/async for real concurrency)
    # start_twitter_stream(TWITTER_BEARER_TOKEN, KEYWORDS, handle_event) # XXX: Uncomment to use Twitter API
    fake_twitter_stream(keywords=KEYWORDS, on_event=handle_event) # XXX: Simulate Twitter stream for testing (Comment out if using Twitter API)
    # start_youtube_scraper(callback=process_event)

    # FOR TESTING: SAVE the graph for manual inspection
    gpickle.write_gpickle(graph.G, "../data/test_graph.gpickle")
    print("Graph saved to data/test_graph.gpickle!")


if __name__ == "__main__":
    main()