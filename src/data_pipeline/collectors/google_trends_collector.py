from datetime import datetime, timezone
import time
import random
import json
from pathlib import Path

from utils.text import slugify as _slugify

#TODO: Implement real Google Trends API integration here when available

class RealisticTrendsSimulator: 
    """Generate realistic trending topics based on time, patterns, and cycles."""
    
    def __init__(self):
        # Time-based trending patterns
        self.hourly_patterns = {
            "morning": (6, 11, ["coffee", "breakfast", "commute", "news", "weather", "workout"]),
            "lunch": (11, 14, ["lunch", "food", "work", "productivity", "midday news"]),
            "afternoon": (14, 17, ["work", "technology", "business", "stock market", "productivity"]),
            "evening": (17, 21, ["dinner", "entertainment", "sports", "streaming", "family time"]),
            "night": (21, 6, ["relaxation", "movies", "gaming", "late night", "social media"])
        }
        
        # Day-of-week patterns
        self.weekly_patterns = {
            0: ["monday motivation", "week ahead", "work preparation"],  # Monday
            1: ["tuesday productivity", "mid-week", "technology news"],   # Tuesday  
            2: ["wednesday wisdom", "hump day", "career advice"],        # Wednesday
            3: ["thursday thoughts", "weekend planning", "entertainment"], # Thursday
            4: ["friday feeling", "weekend prep", "social plans"],       # Friday
            5: ["saturday fun", "weekend activities", "leisure"],        # Saturday
            6: ["sunday relaxation", "week reflection", "preparation"]   # Sunday
        }
        
        # Seasonal/cyclical trends
        self.seasonal_trends = {
            "winter": ["holiday shopping", "winter fashion", "flu season", "new year resolutions"],
            "spring": ["spring cleaning", "gardening", "allergy season", "easter"],
            "summer": ["vacation planning", "summer fashion", "outdoor activities", "festivals"],
            "fall": ["back to school", "fall fashion", "halloween", "thanksgiving"]
        }
        
        # Evergreen trending categories
        self.evergreen_categories = {
            "technology": ["iPhone", "AI", "electric cars", "social media", "crypto", "gaming"],
            "entertainment": ["movies", "TV shows", "music", "celebrities", "streaming", "awards"],
            "health": ["fitness", "diet", "mental health", "wellness", "medical news"],
            "finance": ["stock market", "investing", "economy", "real estate", "taxes"],
            "lifestyle": ["travel", "food", "fashion", "home decor", "relationships"],
            "sports": ["football", "basketball", "soccer", "olympics", "championships"],
            "news": ["politics", "world events", "breaking news", "elections", "climate"]
        }
        
        # Current "hot" topics that cycle in and out
        self.current_hot_topics = [
            "climate change", "artificial intelligence", "electric vehicles", 
            "remote work", "inflation", "housing market", "space exploration",
            "renewable energy", "cybersecurity", "mental health awareness"
        ]
        
        # Viral/meme patterns (short-lived, high intensity)
        self.viral_patterns = [
            "viral video", "tiktok trend", "meme", "social media challenge",
            "celebrity news", "scandal", "breakthrough", "phenomenon"
        ]
    
    def get_current_trends(self, count: int = 8) -> list:
        """Generate realistic trends for the current time."""
        now = datetime.now()
        trends = []
        
        # 1. Time-based trends (40% of trends)
        time_trends = self._get_time_based_trends(now, int(count * 0.4))
        trends.extend(time_trends)
        
        # 2. Category-based evergreen trends (30% of trends)  
        category_trends = self._get_category_trends(int(count * 0.3))
        trends.extend(category_trends)
        
        # 3. Hot topics (20% of trends)
        hot_trends = self._get_hot_topics(int(count * 0.2))
        trends.extend(hot_trends)
        
        # 4. Viral/random (10% of trends)
        viral_trends = self._get_viral_trends(max(1, count - len(trends)))
        trends.extend(viral_trends)
        
        # Ensure we have exactly the requested count
        if len(trends) > count:
            trends = random.sample(trends, count)
        elif len(trends) < count:
            # Fill remaining with random evergreen
            remaining = count - len(trends)
            all_evergreen = [item for category in self.evergreen_categories.values() for item in category]
            trends.extend(random.sample(all_evergreen, min(remaining, len(all_evergreen))))
        
        return trends[:count]
    
    def _get_time_based_trends(self, now: datetime, count: int) -> list:
        """Get trends based on current time of day and week."""
        trends = []
        hour = now.hour
        weekday = now.weekday()
        
        # Hour-based trends
        for period, (start_hour, end_hour, topics) in self.hourly_patterns.items():
            if start_hour <= hour < end_hour or (start_hour > end_hour and (hour >= start_hour or hour < end_hour)):
                trends.extend(random.sample(topics, min(2, len(topics))))
                break
        
        # Day-of-week trends
        if weekday in self.weekly_patterns:
            trends.extend(random.sample(self.weekly_patterns[weekday], min(2, len(self.weekly_patterns[weekday]))))
        
        return random.sample(trends, min(count, len(trends))) if trends else []
    
    def _get_category_trends(self, count: int) -> list:
        """Get trends from evergreen categories."""
        trends = []
        categories = random.sample(list(self.evergreen_categories.keys()), min(3, len(self.evergreen_categories)))
        
        for category in categories:
            items = self.evergreen_categories[category]
            trends.extend(random.sample(items, min(2, len(items))))
        
        return random.sample(trends, min(count, len(trends))) if trends else []
    
    def _get_hot_topics(self, count: int) -> list:
        """Get current hot topics."""
        return random.sample(self.current_hot_topics, min(count, len(self.current_hot_topics)))
    
    def _get_viral_trends(self, count: int) -> list:
        """Get viral/meme-style trends."""
        viral_topics = []
        for pattern in random.sample(self.viral_patterns, min(2, len(self.viral_patterns))):
            # Add some randomness to viral topics
            variations = [
                f"latest {pattern}",
                f"{pattern} 2024", 
                f"trending {pattern}",
                f"viral {pattern}",
                pattern
            ]
            viral_topics.extend(random.sample(variations, 1))
        
        return viral_topics[:count]


def start_google_trends_collector(
    on_event=None,
    region: str = "US",
    category: str = "all", 
    count: int = 20,
    interval: int = 3600,
):
    """Enhanced simulated Google Trends collector."""
    print("[Simulated Google Trends Collector] Starting realistic trend simulation...")
    
    simulator = RealisticTrendsSimulator()
    
    while True:
        try:
            # Generate realistic trends for current time
            trending_topics = simulator.get_current_trends(count=min(count, 12))
            
            print(f"[Simulated Trends] Generated {len(trending_topics)} realistic trends")
            
            for topic in trending_topics:
                # Add realistic variations and context
                variations = [
                    f"{topic} 2024",
                    f"latest {topic}",
                    f"{topic} news", 
                    f"trending {topic}",
                    f"best {topic}",
                    topic
                ]
                
                final_topic = random.choice(variations)
                
                # Generate realistic context
                contexts = ["trending", "popular"]
                if "news" in topic.lower():
                    contexts.extend(["breaking", "latest", "developing"])
                if "tech" in topic.lower() or "AI" in topic:
                    contexts.extend(["technology", "innovation", "future"])
                if any(sport in topic.lower() for sport in ["football", "basketball", "soccer"]):
                    contexts.extend(["sports", "championship", "league"])
                
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content_id": f"trend_{_slugify(final_topic)}",
                    "source": "google_trends",
                    "type": "trend",
                    "text": final_topic,
                    "context": random.sample(contexts, min(3, len(contexts))),
                    "region": region,
                    "simulated": True  # Mark as simulated for transparency
                }
                
                if on_event:
                    on_event(event)
                else:
                    print(f"  📈 {final_topic}")
                    
                # Small delay between events
                time.sleep(1)
            
            print(f"[Simulated Trends] Waiting {interval} seconds for next trend cycle...")
            time.sleep(interval)
            
        except Exception as e:
            print(f"[Simulated Trends] Error generating trends: {e}")
            time.sleep(60)


def fake_google_trends_stream(on_event=None, n_events: int = 5, delay: float = 1.0):
    """Simple fake trends for quick testing (legacy compatibility)."""
    simulator = RealisticTrendsSimulator()
    trending_topics = simulator.get_current_trends(count=n_events)
    
    print(f"[Fake Trends] Generating {n_events} quick test trends...")
    
    for topic in trending_topics:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_id": f"trend_{_slugify(topic)}",
            "source": "google_trends", 
            "type": "trend",
            "text": topic,
            "context": ["test", "simulation"],
            "simulated": True
        }
        
        if on_event:
            on_event(event)
        else:
            print(f"  📈 {topic}")
            
        time.sleep(delay)
    
    print("[Fake Trends] Finished generating test trends")