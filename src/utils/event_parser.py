def parse_event(event):
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
        # Context/trend nodes (e.g. for trend-topic or breaking event)
        outputs.append(("trend-event", None, content, "trend"))

    return outputs