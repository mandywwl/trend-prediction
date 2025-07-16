def parse_event(event):
    ts = event["timestamp"]
    u = event.get("user_id")
    t = event.get("tweet_id")
    hashtags = event.get("hashtags", [])
    
    if event["type"] == "original":
        outputs = [("user-posted", u, t, "posted")]
        for tag in hashtags:
            outputs.append((None, t, f"h_{tag}", "hashtagged"))
        return outputs[0]  # One edge per event for simplicity
    elif event["type"] == "retweet":
        return ("user-retweet", u, t, "retweeted")
    elif event["type"] == "like":
        return ("user-like", u, t, "liked")
    else:
        return (None, None, None, None)