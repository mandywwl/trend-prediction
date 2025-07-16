from collectors.twitter_collector import start_twitter_stream
#from collectors.youtube_collector import start_youtube_scraper
from graph_builder import GraphBuilder

graph = GraphBuilder()

def main():
    # For each platform, define a callback to process events
    def process_event(event):
        graph.process_event(event)
    
    # Start collectors (could be threaded/async for real concurrency)
    start_twitter_stream(callback=process_event)
    start_youtube_scraper(callback=process_event)
    # ...repeat for TikTok, Google Trends

if __name__ == "__main__":
    main()
