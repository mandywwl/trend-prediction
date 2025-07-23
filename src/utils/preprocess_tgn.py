import json
import numpy as np
from datetime import datetime

# --- Utility functions for TGN preprocessing ---
node2id = {}
def get_node_id(node):
    if node not in node2id:
        node2id[node] = len(node2id)
    return node2id[node]

# Convert ISO8601 timestamp to float (seconds since epoch)
def to_timestamp(ts):
    return datetime.fromisoformat(ts).timestamp()

src_nodes, dst_nodes, timestamps, edge_features = [], [], [], []

# --- Read events from JSONL file and build edges ---
with open('../data/events.jsonl') as f:
    for line in f:
        event = json.loads(line)
        user = get_node_id(event['user_id'])
        content = get_node_id(event['content_id'])
        time_val = to_timestamp(event['timestamp'])
        platform = event['source']
        e_type = event['type']

        # One-hot: [is_twitter, is_youtube, is_original, is_retweet, is_upload]
        feature = [
            int(platform == "twitter"),
            int(platform == "youtube"),
            int(e_type == "original"),
            int(e_type == "retweet"),
            int(e_type == "upload")
        ]

        # Add edge: user → content
        src_nodes.append(user)
        dst_nodes.append(content)
        timestamps.append(time_val)
        edge_features.append(feature)

        for hashtag in event['hashtags']:
            h_id = get_node_id("h_" + hashtag)
            src_nodes.append(content)
            dst_nodes.append(h_id)
            timestamps.append(time_val)
            edge_features.append([0, 0, 0, 0, 0])  # Placeholder, can improve later

# Save to .npz for easy loading in PyTorch
np.savez('../data/tgn_edges_basic.npz',
         src=np.array(src_nodes),
         dst=np.array(dst_nodes),
         t=np.array(timestamps),
         edge_attr=np.array(edge_features),
         node_map=np.array(list(node2id.keys())))

print(f"Saved {len(src_nodes)} edges and {len(node2id)} nodes.")
