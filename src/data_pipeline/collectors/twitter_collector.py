"""Twitter/X data collector using Tweepy v2 API with enhanced realistic simulation."""

import tweepy
import time
import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List

from .base import BaseCollector, SimulatedCollector
from utils.logging import get_logger

logger = get_logger(__name__)

class TwitterCollector(tweepy.StreamingClient, BaseCollector):
    """Twitter/X Streaming Collector using Tweepy v2.
    
    Ingests tweets in real-time and standardizes events for processing.
    """
    

    def __init__(
        self, 
        bearer_token: str, 
        keywords: Optional[List[str]] = None, 
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize Twitter collector.
        
        Args:
            bearer_token: Twitter/X API bearer token
            keywords: List of keywords or hashtags to track (default: all)
            on_event: Callback function to handle each processed event
        """
        tweepy.StreamingClient.__init__(self, bearer_token)
        BaseCollector.__init__(self, "twitter", on_event)
        
        self.keywords = keywords or []

        # Add stream rules if provided
        if self.keywords:
            self.add_rules(tweepy.StreamRule(" OR ".join(self.keywords)))

    def on_tweet(self, tweet):
        """Called for every new tweet event matching the stream rules."""
        try:
            # Standardize event schema
            event_data = {
                "user_id": f"u{tweet.author_id}",
                "content_id": f"t{tweet.id}",
                "hashtags": (
                    [tag["tag"].lower() for tag in tweet.entities.get("hashtags", [])]
                    if tweet.entities
                    else []
                ),
                "type": (
                    "original"
                    if not tweet.referenced_tweets
                    else tweet.referenced_tweets[0].type
                ),
                "text": tweet.text,
            }

            event = self._create_base_event(**event_data)
            self._emit_event(event)
            
            logger.info(f"Received tweet {event['content_id']} from {event['user_id']}")
            
        except Exception as e:
            self._handle_error(e, "processing tweet")

    def on_errors(self, errors):
        """Handle streaming errors."""
        self._handle_error(Exception(f"Stream errors: {errors}"), "streaming")

    def on_connection_error(self):
        """Handle connection errors."""
        logger.warning("Connection error. Restarting...")
        self.disconnect()
        
    def start(self) -> None:
        """Start the Twitter stream."""
        logger.info("Starting Twitter stream...")
        self.is_running = True
        self.filter(
            tweet_fields=["created_at", "author_id", "entities", "referenced_tweets"],
            expansions=[],
            threaded=False,
        )
        
    def stop(self) -> None:
        """Stop the Twitter stream."""
        self.is_running = False
        self.disconnect()

class RealisticTweetSimulator:
    """Generate realistic tweet content and patterns."""
    
    def __init__(self):
        # Tweet templates by category
        self.tweet_templates = {
            "trending": [
                "Just watched {topic} and it's amazing! #{hashtag} #trending",
                "Can't stop thinking about {topic} 🔥 #{hashtag}",
                "Everyone's talking about {topic} but honestly... #{hashtag} #unpopularopinion",
                "{topic} is everywhere right now! What do you think? #{hashtag}",
                "Hot take: {topic} is overrated. Fight me. #{hashtag} #hottake"
            ],
            "technology": [
                "The new {topic} update is actually pretty good! #{hashtag} #tech",
                "Why is {topic} so complicated? Just me? #{hashtag} #techproblems",
                "Finally got my hands on {topic}! First impressions thread 🧵 #{hashtag}",
                "{topic} changed my workflow completely. Highly recommend! #{hashtag} #productivity",
                "Is anyone else having issues with {topic}? #{hashtag} #help"
            ],
            "entertainment": [
                "Just finished {topic} and I'm emotionally destroyed 😭 #{hashtag}",
                "{topic} soundtrack is pure fire 🎵 #{hashtag} #music",
                "Unpopular opinion: {topic} wasn't that good #{hashtag} #controversial",
                "Can't wait for {topic} season 2! When is it coming? #{hashtag}",
                "{topic} had me on the edge of my seat! #{hashtag} #bingewatching"
            ],
            "lifestyle": [
                "Starting my {topic} journey today! Wish me luck 💪 #{hashtag}",
                "Day 30 of {topic} and I'm already seeing results! #{hashtag} #transformation",
                "Why didn't anyone tell me {topic} was this hard? #{hashtag} #struggle",
                "{topic} is my new obsession! Anyone else? #{hashtag} #lifestyle",
                "Best {topic} tips? Drop them below! #{hashtag} #help"
            ],
            "news": [
                "Breaking: {topic} just happened. Thoughts? #{hashtag} #breaking",
                "The {topic} situation is getting serious #{hashtag} #news",
                "Media coverage of {topic} is wild #{hashtag} #media",
                "Everyone needs to know about {topic} #{hashtag} #important",
                "Local news: {topic} affecting our community #{hashtag} #local"
            ],
            "sports": [
                "GOAL! {topic} is on fire tonight! ⚽ #{hashtag} #sports",
                "{topic} performance was legendary! #{hashtag} #GOAT",
                "Refs completely missed that call in {topic} game #{hashtag} #controversial",
                "Season stats for {topic} are insane! #{hashtag} #analytics",
                "Trade rumors: {topic} might be moving! #{hashtag} #rumors"
            ],
            "random": [
                "Coffee thought: {topic} makes everything better ☕ #{hashtag}",
                "Shower thought: What if {topic} but for {random_topic}? #{hashtag} #showerthoughts",
                "PSA: {topic} is not a personality trait #{hashtag} #truth",
                "Me: I won't get obsessed with {topic}\nAlso me: #{hashtag} #relatable",
                "That feeling when {topic} hits different #{hashtag} #mood"
            ]
        }
        
        # Popular hashtags by category
        self.hashtag_pools = {
            "trending": ["viral", "trending", "fyp", "mood", "relatable", "facts"],
            "technology": ["tech", "innovation", "AI", "coding", "startup", "digital"],
            "entertainment": ["movies", "tv", "music", "celebrity", "drama", "review"],
            "lifestyle": ["health", "fitness", "food", "travel", "fashion", "selfcare"],
            "news": ["breaking", "politics", "world", "local", "update", "important"],
            "sports": ["sports", "game", "championship", "team", "player", "stats"],
            "random": ["random", "thoughts", "life", "mood", "real", "truth"]
        }
        
        # Time-based posting patterns
        self.time_patterns = {
            "morning": (6, 11, ["coffee", "commute", "work", "motivation", "news"]),
            "lunch": (11, 14, ["lunch", "food", "break", "tired", "work"]),
            "afternoon": (14, 17, ["work", "productivity", "tired", "almost done"]),
            "evening": (17, 21, ["dinner", "home", "relax", "family", "entertainment"]),
            "night": (21, 6, ["netflix", "gaming", "insomnia", "late night", "tomorrow"])
        }
        
        # User personas with different posting styles
        self.user_personas = {
            "tech_enthusiast": {
                "bio": "Software developer, AI enthusiast",
                "preferred_categories": ["technology", "trending"],
                "posting_frequency": "high",
                "hashtag_style": "tech-heavy"
            },
            "entertainment_fan": {
                "bio": "Pop culture addict, binge watcher",
                "preferred_categories": ["entertainment", "trending"],
                "posting_frequency": "medium",
                "hashtag_style": "pop-culture"
            },
            "news_junkie": {
                "bio": "Keeping up with current events",
                "preferred_categories": ["news", "trending"],
                "posting_frequency": "medium",
                "hashtag_style": "serious"
            },
            "lifestyle_blogger": {
                "bio": "Health, wellness, and good vibes",
                "preferred_categories": ["lifestyle", "random"],
                "posting_frequency": "low",
                "hashtag_style": "lifestyle"
            },
            "sports_fan": {
                "bio": "Die-hard sports enthusiast",
                "preferred_categories": ["sports", "trending"],
                "posting_frequency": "high",
                "hashtag_style": "sports"
            },
            "casual_user": {
                "bio": "Just here for the memes",
                "preferred_categories": ["random", "trending"],
                "posting_frequency": "low",
                "hashtag_style": "casual"
            }
        }
    
    def generate_realistic_tweet(self, user_persona: str = None, topic_hint: str = None) -> Dict[str, Any]:
        """Generate a realistic tweet with proper timing and content."""
        # Select user persona
        if not user_persona:
            user_persona = random.choice(list(self.user_personas.keys()))
        
        persona = self.user_personas[user_persona]
        
        # Choose category based on persona and time
        hour = datetime.now().hour
        time_context = self._get_time_context(hour)
        
        if topic_hint:
            # Use hint to influence category selection
            category = self._categorize_topic(topic_hint)
        else:
            # Choose from persona's preferred categories
            category = random.choice(persona["preferred_categories"])
        
        # Generate tweet content
        template = random.choice(self.tweet_templates[category])
        
        # Fill in template
        if topic_hint:
            topic = topic_hint
        else:
            topic = self._generate_topic_for_category(category, time_context)
        
        # Create hashtags
        hashtags = self._generate_hashtags(category, persona["hashtag_style"])
        
        # Replace placeholders
        tweet_text = template.format(
            topic=topic,
            hashtag=hashtags[0] if hashtags else "trending",
            random_topic=random.choice(["cats", "pizza", "monday", "coffee", "wifi"])
        )
        
        return {
            "text": tweet_text,
            "hashtags": hashtags,
            "persona": user_persona,
            "category": category,
            "time_context": time_context
        }
    
    def _get_time_context(self, hour: int) -> str:
        """Get time context for current hour."""
        for period, (start, end, _) in self.time_patterns.items():
            if start <= hour < end or (start > end and (hour >= start or hour < end)):
                return period
        return "random"
    
    def _categorize_topic(self, topic: str) -> str:
        """Categorize a topic hint into appropriate category."""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["ai", "tech", "software", "app", "coding"]):
            return "technology"
        elif any(word in topic_lower for word in ["movie", "show", "music", "celebrity", "entertainment"]):
            return "entertainment"
        elif any(word in topic_lower for word in ["health", "fitness", "food", "travel", "lifestyle"]):
            return "lifestyle"
        elif any(word in topic_lower for word in ["news", "politics", "breaking", "election"]):
            return "news"
        elif any(word in topic_lower for word in ["sports", "game", "team", "player", "championship"]):
            return "sports"
        else:
            return "trending"
    
    def _generate_topic_for_category(self, category: str, time_context: str) -> str:
        """Generate a topic appropriate for category and time."""
        topics_by_category = {
            "technology": ["iPhone 16", "ChatGPT", "Tesla", "Google AI", "coding bootcamp"],
            "entertainment": ["Marvel movie", "Netflix series", "Taylor Swift", "Oscar nominations", "viral TikTok"],
            "lifestyle": ["morning routine", "healthy recipes", "workout plan", "meditation", "sustainable living"],
            "news": ["election update", "climate summit", "economic report", "breaking news", "local politics"],
            "sports": ["NBA playoffs", "World Cup", "Super Bowl", "trade deadline", "rookie season"],
            "trending": ["viral meme", "Twitter drama", "celebrity gossip", "internet challenge", "controversial take"]
        }
        
        # Add time-based topics
        if time_context in self.time_patterns:
            time_topics = self.time_patterns[time_context][2]
            if random.random() < 0.3:  # 30% chance to use time-based topic
                return random.choice(time_topics)
        
        return random.choice(topics_by_category.get(category, topics_by_category["trending"]))
    
    def _generate_hashtags(self, category: str, style: str) -> List[str]:
        """Generate appropriate hashtags for category and style."""
        base_hashtags = self.hashtag_pools.get(category, ["trending"])
        
        # Adjust number of hashtags based on style
        if style == "tech-heavy":
            num_hashtags = random.randint(2, 4)
        elif style == "pop-culture":
            num_hashtags = random.randint(3, 5)
        elif style == "serious":
            num_hashtags = random.randint(1, 2)
        else:
            num_hashtags = random.randint(1, 3)
        
        # Select hashtags
        selected = random.sample(base_hashtags, min(num_hashtags, len(base_hashtags)))
        
        # Add some randomness
        if random.random() < 0.3:  # 30% chance to add cross-category hashtag
            other_categories = [cat for cat in self.hashtag_pools.keys() if cat != category]
            other_category = random.choice(other_categories)
            selected.append(random.choice(self.hashtag_pools[other_category]))
        
        return selected





class EnhancedFakeTwitterCollector(SimulatedCollector):
    """Enhanced simulated Twitter collector with realistic content and patterns."""
    
    def __init__(
        self, 
        keywords: Optional[List[str]] = None, 
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize enhanced fake Twitter collector.
        
        Args:
            keywords: Keywords to influence tweet content generation
            on_event: Callback function to handle events
        """
        super().__init__("twitter", on_event)
        self.keywords = keywords or []
        self.simulator = RealisticTweetSimulator()
        
        # Create persistent user personas
        self.users = {}
        for i in range(10):  # 10 fake users
            persona = random.choice(list(self.simulator.user_personas.keys()))
            self.users[f"u{i}"] = {
                "persona": persona,
                "tweet_count": 0,
                "last_tweet_time": None
            }
    
    def simulate_continuous_stream(
        self, 
        events_per_batch: int = 5, 
        batch_interval: int = 180,
        topic_hints: Optional[List[str]] = None
    ) -> None:
        """Simulate continuous Twitter stream with realistic timing and content.
        
        Args:
            events_per_batch: Number of tweets per batch
            batch_interval: Seconds between batches
            topic_hints: Optional list of trending topics to influence content
        """
        logger.info(f"[Enhanced Twitter] Starting continuous stream simulation")
        
        while not self.shutdown_event.is_set() if hasattr(self, 'shutdown_event') else True:
            try:
                # Generate batch of realistic tweets
                for i in range(events_per_batch):
                    # Select user
                    user_id = f"u{random.randint(0, 9)}"
                    user_info = self.users[user_id]
                    
                    # Choose topic hint if available
                    topic_hint = None
                    if topic_hints and random.random() < 0.4:  # 40% chance to use trending topic
                        topic_hint = random.choice(topic_hints)
                    elif self.keywords and random.random() < 0.3:  # 30% chance to use keywords
                        topic_hint = random.choice(self.keywords)
                    
                    # Generate realistic tweet
                    tweet_data = self.simulator.generate_realistic_tweet(
                        user_persona=user_info["persona"],
                        topic_hint=topic_hint
                    )
                    
                    # Create event
                    event_data = {
                        "user_id": user_id,
                        "content_id": f"t{user_info['tweet_count']}_{user_id}",
                        "hashtags": tweet_data["hashtags"],
                        "type": "original",
                        "text": tweet_data["text"],
                        "persona": tweet_data["persona"],
                        "category": tweet_data["category"],
                        "simulated": True
                    }
                    
                    event = self._create_base_event(**event_data)
                    self._emit_event(event)
                    
                    # Update user stats
                    user_info["tweet_count"] += 1
                    user_info["last_tweet_time"] = datetime.now()
                    
                    # Small delay between tweets in batch
                    time.sleep(random.uniform(1, 3))
                
                logger.info(f"[Enhanced Twitter] Generated {events_per_batch} realistic tweets")
                
                # Wait for next batch
                if hasattr(self, 'shutdown_event') and self.shutdown_event.is_set():
                    break
                
                time.sleep(batch_interval)
                
            except Exception as e:
                logger.error(f"[Enhanced Twitter] Error in continuous stream: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _generate_default_event(self, index: int) -> Dict[str, Any]:
        """Generate a realistic simulated Twitter event (legacy compatibility)."""
        user_id = f"u{index % 10}"
        
        # Get or create user
        if user_id not in self.users:
            persona = random.choice(list(self.simulator.user_personas.keys()))
            self.users[user_id] = {"persona": persona, "tweet_count": 0}
        
        user_info = self.users[user_id]
        
        # Generate realistic tweet
        tweet_data = self.simulator.generate_realistic_tweet(
            user_persona=user_info["persona"]
        )
        
        return {
            "user_id": user_id,
            "content_id": f"t{index}",
            "hashtags": tweet_data["hashtags"],
            "type": "original",
            "text": tweet_data["text"],
            "simulated": True
        }

# API Functions
def start_twitter_stream(
    bearer_token: str, 
    keywords: Optional[List[str]] = None, 
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
) -> TwitterCollector:
    """Entry point for launching the Twitter stream collector.
    
    Args:
        bearer_token: Twitter API bearer token
        keywords: Keywords to track
        on_event: Event callback function
        
    Returns:
        TwitterCollector instance
    """
    collector = TwitterCollector(
        bearer_token=bearer_token, 
        keywords=keywords, 
        on_event=on_event
    )
    collector.start()
    return collector


def enhanced_fake_twitter_stream(
    keywords: Optional[List[str]] = None, 
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None, 
    events_per_batch: int = 5,
    batch_interval: int = 180,
    topic_hints: Optional[List[str]] = None
) -> None:
    """Generate continuous realistic Twitter stream simulation.
    
    Args:
        keywords: Keywords to influence tweet content
        on_event: Callback function to process each event
        events_per_batch: Number of tweets per batch
        batch_interval: Seconds between batches
        topic_hints: Optional trending topics to influence content
    """
    collector = EnhancedFakeTwitterCollector(keywords=keywords, on_event=on_event)
    collector.simulate_continuous_stream(
        events_per_batch=events_per_batch,
        batch_interval=batch_interval,
        topic_hints=topic_hints
    )



def realistic_fake_twitter_stream(
    keywords: Optional[List[str]] = None, 
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None, 
    n_events: int = 10, 
    delay: float = 1.0,
    topic_hints: Optional[List[str]] = None
) -> None:
    """Generate realistic fake Twitter events for testing.
    
    Args:
        keywords: Keywords to influence content
        on_event: Callback function to process each event
        n_events: Number of events to generate
        delay: Seconds between events
        topic_hints: Optional trending topics to influence content
    """
    collector = EnhancedFakeTwitterCollector(keywords=keywords, on_event=on_event)
    
    # Set topic hints for the simulator
    if topic_hints:
        for i in range(n_events):
            topic_hint = random.choice(topic_hints) if random.random() < 0.5 else None
            user_id = f"u{i % 10}"
            
            if user_id not in collector.users:
                persona = random.choice(list(collector.simulator.user_personas.keys()))
                collector.users[user_id] = {"persona": persona, "tweet_count": i}
            
            tweet_data = collector.simulator.generate_realistic_tweet(
                user_persona=collector.users[user_id]["persona"],
                topic_hint=topic_hint
            )
            
            event_data = {
                "user_id": user_id,
                "content_id": f"t{i}",
                "hashtags": tweet_data["hashtags"],
                "type": "original",
                "text": tweet_data["text"],
                "simulated": True
            }
            
            event = collector._create_base_event(**event_data)
            collector._emit_event(event)
            
            time.sleep(delay)
    else:
        # Use default simulation
        collector.simulate_events(n_events=n_events, delay=delay)