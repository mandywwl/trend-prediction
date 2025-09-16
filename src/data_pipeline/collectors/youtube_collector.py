from googleapiclient.discovery import build
from datetime import datetime, timezone
import time
from config.config import REGIONS
from threading import Event


def start_youtube_api_collector(
    api_key, 
    on_event=None, 
    categories=None, 
    max_results=50, 
    region_code= REGIONS[0], #TODO: Multi-region support
    delay=1.0,
    refresh_interval: int = 900, # 15 minutes
    shutdown_event: Event | None=None,
):
    """
    Poll the YouTube API repeatedly and push items via `on_event(event)`.
    """
    print("[YouTube Collector] Starting stream...")

    if categories is None:
         # Default categories: Music, Sports, Gaming, Entertainment, News & Politics, Science & Technology 
        categories = [ "10", "17", "20", "24", "25", "28",] 

    youtube = build("youtube", "v3", developerKey=api_key)

    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            print("[YouTube Collector] Shutdown signal received.")
            break

    
    for category in categories:
        if shutdown_event is not None and shutdown_event.is_set():
            break

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
            if shutdown_event is not None and shutdown_event.is_set():
                break
            try:
                video_id = item["id"]
                channel_title = item["snippet"]["channelTitle"]
                video_title = item["snippet"]["title"]
                tags = item["snippet"].get("tags", [])
                hashtags = [t for t in tags if t.startswith("#")]
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_id": f"yt_u_{channel_title.replace(' ', '_')}",
                    "content_id": f"yt_v_{video_id}",
                    "hashtags": hashtags,
                    "type": "upload",
                    "source": "youtube",
                    "text": video_title,
                }

                if on_event:
                    on_event(event)
                else:
                    print(event)

                # Respect API rate limits + be nice to CPU
                time.sleep(delay)  
            except Exception as e:
                print(f"[YouTube Collector] Error processing item {item.get('id', 'unknown')}: {e}")
                continue

    # Backoff between polling cycles, but remain responsive to shutdown
    slept = 0
    while slept < refresh_interval:
        if shutdown_event is not None and shutdown_event.is_set():
            break
        time.sleep(1)
        slept += 1
