from collectors.twitter_collector import start_twitter_stream
from collectors.twitter_collector import fake_twitter_stream
# from collectors.youtube_collector import start_youtube_scraper
from graph_builder import GraphBuilder

graph = GraphBuilder()

TWITTER_BEARER_TOKEN = "Your_Twitter_Bearer_Token" # XXX:Replace with your actual Twitter/X Bearer Token (Paid versions only as of 2025)
KEYWORDS = ["#trending", "fyp", "viral"]  # XXX: Adjust keywords as needed

# For each platform, define a callback to process events
def handle_event(event):
    graph.process_event(event) # Event comes from any collector, any platform
    print(f"Event from {event['source']}: {event['type']}, updated graph.")
    # TODO: Add optional features: save snapshots, stats, error handling, etc.

def main():

    # Start collectors (could be threaded/async for real concurrency)
    # start_twitter_stream(TWITTER_BEARER_TOKEN, KEYWORDS, handle_event) # XXX: Uncomment to use Twitter API
    fake_twitter_stream(keywords=KEYWORDS, on_event=handle_event) # XXX: Simulate Twitter stream for testing (Comment out if using Twitter API)
    # start_youtube_scraper(callback=process_event)
    

if __name__ == "__main__":
    main()