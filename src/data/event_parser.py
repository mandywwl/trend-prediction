import re

def _slugify(text: str) -> str:
    """Utility: convert free text into a safe identifier."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_event(event):
    """Parse a raw event into structured components for graph processing.
    Standardises identifiers and timestamps across platforms (user IDs, tweet IDs, video IDs).
    """
    source = event.get("source")
    event_type = event.get("type")
    user = event.get("user_id")
    content = event.get("content_id")  # "content_id" for generality
    hashtags = event.get("hashtags", [])
    outputs = []

    if source == "twitter":
        if event_type == "original":
            outputs.append(("user-posted", user, content, "posted"))
            for tag in hashtags:
                outputs.append((None, content, f"h_{tag}", "hashtagged"))
        elif event_type == "retweet":
            outputs.append(("user-retweet", user, content, "retweeted"))
        elif event_type == "like":
            outputs.append(("user-like", user, content, "liked"))
    
    elif source == "youtube":
        if event_type == "upload":
            outputs.append(("user-uploaded", user, content, "uploaded"))
        elif event_type == "comment":
            outputs.append(("user-commented", user, content, "commented"))
        elif event_type == "view":
            outputs.append(("user-viewed", user, content, "viewed"))

    elif source == "tiktok":
        if event_type == "upload":
            outputs.append(("user-uploaded", user, content, "uploaded"))
        elif event_type == "like":
            outputs.append(("user-liked", user, content, "liked"))
        elif event_type == "comment":
            outputs.append(("user-commented", user, content, "commented"))

    elif source == "google_trends":
        # Trend node itself
        outputs.append(("trend-event", None, content, "trend"))
        # Optional context nodes for disambiguation
        for ctx in event.get("context", []):
            ctx_id = f"ctx_{_slugify(ctx)}"
            outputs.append(("trend-context", content, ctx_id, "has_context"))

    return outputs