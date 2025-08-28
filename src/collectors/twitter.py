import tweepy
import time
from datetime import datetime

class TwitterCollector(tweepy.StreamingClient):
    """
    Twitter/X Streaming Collector using Tweepy v2.
    Ingests tweets in real-time and standardizes events for processing.
    """
    def __init__(self, bearer_token, keywords=None, on_event=None):
        """
        :param bearer_token: Twitter/X API bearer token
        :param keywords: List of keywords or hashtags to track (default: all)
        :param on_event: Callback function to handle each processed event (event_dict)
        """
        super().__init__(bearer_token)
        self.keywords = keywords or []
        self.on_event = on_event  # function to call for every parsed event

        # Add stream rules if provided
        if self.keywords:
            self.add_rules(tweepy.StreamRule(" OR ".join(self.keywords)))

    def on_tweet(self, tweet):
        """
        Called for every new tweet event matching the stream rules.
        """
        # Standardize event schema
        event = {
            "timestamp": tweet.created_at.isoformat(),
            "user_id": f"u{tweet.author_id}",
            "content_id": f"t{tweet.id}",
            "hashtags": [tag['tag'].lower() for tag in tweet.entities.get('hashtags', [])] if tweet.entities else [],
            "type": "original" if not tweet.referenced_tweets else tweet.referenced_tweets[0].type, # can be 'retweeted', 'replied_to', etc.
            "source": "twitter",
            "text": tweet.text
        }

        print(f"[Twitter Stream] Received tweet {event['content_id']} from {event['user_id']} at {event['timestamp']}")

        # Pass the event to the graph builder or any handler
        if self.on_event:
            self.on_event(event)
        else:
            print(event)
        
    def on_errors(self, errors):
        print(f"Stream error: {errors}")

    def on_connection_error(self):
        print("Connection error. Restarting...")
        self.disconnect()


def start_twitter_stream(bearer_token, keywords=None, on_event=None):
    """
    Entry point for launching the Twitter stream collector.
    """
    collector = TwitterCollector(
        bearer_token=bearer_token,
        keywords=keywords,
        on_event=on_event
    )
    print("[Twitter Stream] Starting stream...")
    collector.filter(
        tweet_fields=["created_at", "author_id", "entities", "referenced_tweets"],
        expansions=[],
        threaded=False
    )

# The following is for a simulate twitter stream for testing purposes as the actual Twitter API as of 2025 requires a paid plan for real-time streaming.
# For the scope of this university project, we can simulate events instead.

def fake_twitter_stream(keywords=None, on_event=None, n_events=10, delay=1.0):
    """
    Simulate streaming Twitter events for development without API access.
    :param keywords: Unused, but kept for API compatibility.
    :param on_event: Callback function to process each event.
    :param n_events: Number of fake events to emit.
    :param delay: Seconds between each event.
    """
    print("[Fake Twitter Stream] Starting simulation...")
    for i in range(n_events):
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": f"u{i%5}",  # 5 fake users
            "content_id": f"t{i}",
            "hashtags": ["test", "mock"] if i % 2 == 0 else ["fyp", "viral"],
            "type": "original",
            "source": "twitter",
            "text": f"This is a simulated tweet event number {i}."
        }
        if on_event:
            on_event(event)
        else:
            print(event)
        time.sleep(delay)
    print("[Fake Twitter Stream] Finished emitting events.")
