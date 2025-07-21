import time
from datetime import datetime
from playwright.sync_api import sync_playwright

def start_youtube_scraper(on_event=None, n_videos=10, delay=10):
    """
    Scrape YouTube trending videos and emit events in standard schema.
    :param on_event: Callback function to process each event.
    :param n_videos: Number of trending videos to process per run.
    :param delay: Seconds to wait between polling YouTube Trending (for repeated runs).
    """
    print("[YouTube Collector] Starting trending scrape...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.youtube.com/feed/trending")
        page.wait_for_selector('ytd-video-renderer', timeout=15000) # Wait for trending videos to load
        videos = page.query_selector_all('ytd-video-renderer')[:n_videos]

        for video in videos:
            try:
                # NOTE: YouTube's HTML structure may change, so selectors might need updates.
                # --- Title & Link ---
                title_elem = video.query_selector('a#video-title')
                video_title = title_elem.inner_text().strip() if title_elem else ""
                video_link = title_elem.get_attribute("href") if title_elem else ""
                if not video_link or "watch?v=" not in video_link:
                    print(f"[YouTube Collector] Skipping non-standard video link: {video_link}")
                    continue
                video_id = video_link.split('v=')[1]
                
                # --- Channel Name ---
                channel_elem = video.query_selector('ytd-channel-name#channel-name a')
                channel_name = channel_elem.inner_text().strip() if channel_elem else ""

                # --- Description (short) ---
                desc_elem = video.query_selector('yt-formatted-string#description-text')
                description = desc_elem.inner_text().strip() if desc_elem else ""

                # --- Optional: Views and Upload Date ---
                metadata_elems = video.query_selector_all('#metadata-line span')
                views = metadata_elems[0].inner_text().strip() if len(metadata_elems) > 0 else ""
                upload_time = metadata_elems[1].inner_text().strip() if len(metadata_elems) > 1 else ""

                # --- Hashtags from title and full description ---
                # Navigate to video page to get full description and tags
                video_page = browser.new_page()
                video_page.goto(f"https://www.youtube.com/watch?v={video_id}")
                video_page.wait_for_timeout(5000)  # Wait for video page to load
                # Expand for full description (if available)
                try:
                    show_more_btn = video_page.query_selector('tp-yt-paper-button#expand')
                    if show_more_btn:
                        show_more_btn.click()
                        time.sleep(1)
                except Exception:
                    pass
                # Scrape hashtags from description
                desc_elem = video_page.query_selector('div#expanded yt-attributed-string')
                description_text = desc_elem.inner_text().strip() if desc_elem else ""
                hashtags = [w for w in description_text.split() if w.startswith('#')]


                # Extract tags (if available)
                hashtags = extract_hashtags(video_title + " " + description)
                
                # Create event in standard schema
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": f"yt_u_{channel_name.replace(' ', '_')}",
                    "content_id": f"yt_v_{video_id}",
                    "hashtags": hashtags,
                    "type": "upload",
                    "source": "youtube",
                    "text": video_title
                }
                if on_event:
                    on_event(event)
                else:
                    print(event)
            except Exception as e:
                print(f"[YouTube Collector] Error parsing video: {e}")

            time.sleep(delay) # Delay between videos to avoid rate limits

        browser.close()
        print("[YouTube Collector] Scraping finished.")

def extract_hashtags(text):
    """Extract hashtags from a string, return as list (e.g., ['#fun', '#viral'])."""
    return [word for word in text.split() if word.startswith("#")]