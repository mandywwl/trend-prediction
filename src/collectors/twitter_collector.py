import tweepy

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
            "tweet_id": f"t{tweet.id}",
            "hashtags": [tag['tag'].lower() for tag in tweet.entities.get('hashtags', [])] if tweet.entities else [],
            "type": "original" if not tweet.referenced_tweets else tweet.referenced_tweets[0].type, # can be 'retweeted', 'replied_to', etc.
            "source": "twitter",
            "text": tweet.text
        }
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
    collector.filter(
        tweet_fields=["created_at", "author_id", "entities", "referenced_tweets"],
        expansions=[],
        threaded=False
    )

# Test usage:
# if __name__ == "__main__":
#     # Put your Twitter Bearer Token here or use an environment variable for security
#     BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN") or "YOUR_BEARER_TOKEN_HERE"
#     KEYWORDS = ["#trending", "ai", "football"]  # adjust as needed

#     def handle_event(event):
#         print("[Twitter Event]", event)
#         # Optionally: pass to your graph builder here

#     start_twitter_stream(BEARER_TOKEN, KEYWORDS, handle_event)