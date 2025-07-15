from datetime import datetime

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

def apply_time_decay(ts, reference_time, decay_factor=0.5):
    if isinstance(reference_time, str):
        reference_time = datetime.fromisoformat(reference_time)
    delta_hours = (ts - reference_time).total_seconds() / 3600.0
    return round(decay_factor ** delta_hours, 4)

def validate_graph(G):
    try:
        assert not any(u == v for u, v in G.edges()), "ERROR: Graph has self-loops!"
        assert all(0 <= d['weight'] <= 1 for _, _, d in G.edges(data=True)), "ERROR: Edge weights out of range!"
        assert all('type' in d for _, d in G.nodes(data=True)), "ERROR: Some nodes lack a type!"
        print("SUCCESS: Graph passed all integrity checks.")
    except AssertionError as e:
        print(str(e))
