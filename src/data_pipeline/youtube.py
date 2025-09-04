from googleapiclient.discovery import build
from datetime import datetime
import time


def start_youtube_api_collector(
    api_key, on_event=None, categories=None, max_results=10, region_code="US", delay=1.0
):
    """
    Start collecting trending YouTube videos using the YouTube Data API.
    :param api_key: YouTube Data API key
    :param on_event: Callback function to handle each processed event (event_dict)
    :param categories: List of video categories to filter (optional)
    :param max_results: Maximum number of results to fetch (default: 10)
    :param region_code: Region code for trending videos (default: "US")
    :param delay: Delay between requests to avoid hitting API limits (default: 1.0 seconds)
    """
    print("[YouTube Collector] Starting stream...")

    if categories is None:
        categories = [
            "10",
            "17",
            "20",
            "24",
            "25",
            "28",
        ]  # Default categories: Music, Sports, Gaming, Entertainment, News & Politics, Science & Technology
    youtube = build("youtube", "v3", developerKey=api_key)
    for category in categories:
        print(f"[YouTube Collector] Fetching trending videos for category: {category}")
        try:
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                chart="mostPopular",
                regionCode=region_code,
                videoCategoryId=category,
                maxResults=max_results,
            )
            response = request.execute()
        except Exception as e:
            print(f"[YouTube Collector] API error for category {category}: {e}")
            continue
        for item in response.get("items", []):
            try:
                video_id = item["id"]
                channel_title = item["snippet"]["channelTitle"]
                video_title = item["snippet"]["title"]
                tags = item["snippet"].get("tags", [])
                hashtags = [t for t in tags if t.startswith("#")]
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": f"yt_u_{channel_title.replace(' ', '_')}",
                    "content_id": f"yt_v_{video_id}",
                    "hashtags": hashtags,
                    "type": "upload",
                    "source": "youtube",
                    "text": video_title,
                    # XXX (Optional fields):
                    # "description": description,
                    # "views": item['statistics'].get('viewCount', None),
                    # "likes": item['statistics'].get('likeCount', None),
                }
                if on_event:
                    on_event(event)
                else:
                    print(event)
                time.sleep(delay)  # Respect API rate limits
            except Exception as e:
                print(f"[YouTube API Collector] Error processing item {video_id}: {e}")
                continue
