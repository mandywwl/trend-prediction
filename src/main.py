from collectors.twitter_collector import start_twitter_stream
# from collectors.youtube_collector import start_youtube_scraper
from graph_builder import GraphBuilder

graph = GraphBuilder()

BEARER_TOKEN = "YOUR_BEARER_TOKEN"
KEYWORDS = ["#trending", "fyp", "viral"]  # TODO: Adjust keywords as needed

# For each platform, define a callback to process events
def handle_event(event):
    graph.process_event(event) # Event comes from any collector, any platform
    print(f"Event from {event['source']}: {event['type']}, updated graph.")
    # TODO: Add optional features: save snapshots, stats, error handling, etc.

def main():
    # Start collectors (could be threaded/async for real concurrency)
    start_twitter_stream(BEARER_TOKEN, KEYWORDS, handle_event)
    # start_youtube_scraper(callback=process_event)

if __name__ == "__main__":
    main()