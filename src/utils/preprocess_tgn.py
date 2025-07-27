import json
import numpy as np
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import os

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

node2id = {}
content_text = {}
node_features = {} # node_id: embedding

# --- Utility functions for TGN preprocessing ---
# Get unique node ID for a given node
def get_node_id(node):
    if node not in node2id:
        node2id[node] = len(node2id)
    return node2id[node]

# Convert ISO8601 timestamp to float (seconds since epoch)
def to_timestamp(ts):
    return datetime.fromisoformat(ts).timestamp()

def embed_text(text):
    """Embed text using DistilBERT and return the mean pooling of the output."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb

src_nodes, dst_nodes, timestamps, edge_features = [], [], [], []

# --- Read events from JSONL file and build edges ---

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
events_path = os.path.join(project_root, 'data', 'events.jsonl')
with open(events_path, encoding='utf-8') as f:
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
            # TODO: Add more platforms after data collector complete
            int(e_type == "original"),
            int(e_type == "retweet"),
            int(e_type == "upload")
        ]

        # Add edge: user — content
        src_nodes.append(user)
        dst_nodes.append(content)
        timestamps.append(time_val)
        edge_features.append(feature)

        # Extract content node embedding ONLY if first seen
        if content not in node_features:
            node_features[content] = embed_text(event['text'])

        # User node (NOTE: simple zeros for now)
        if user not in node_features:
            node_features[user] = np.zeros(768)  # DistilBERT base output size

        # Hashtag nodes
        for hashtag in event['hashtags']:
            h_id = get_node_id("h_" + hashtag)
            src_nodes.append(content)
            dst_nodes.append(h_id)
            timestamps.append(time_val)
            edge_features.append([0, 0, 0, 0, 0])  # NOTE: Placeholder, can improve later

# Save to .npz for easy loading in PyTorch
np.savez('../data/tgn_edges_basic.npz',
         src=np.array(src_nodes),
         dst=np.array(dst_nodes),
         t=np.array(timestamps),
         edge_attr=np.array(edge_features),
         node_map=np.array(list(node2id.keys())))

print(f"Saved {len(src_nodes)} edges and {len(node2id)} nodes.")
