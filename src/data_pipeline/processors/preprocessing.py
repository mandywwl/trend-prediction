import json
import numpy as np
import torch
from pathlib import Path

from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertModel

from utils.path_utils import find_repo_root


def build_tgn(
    events_path: Path | str | None = None,
    output_path: Path | str | None = None,
    force: bool = False,
    max_text_len: int = 32,
) -> Path:
    """Build TGN edge file from events.jsonl."""
    repo_root = find_repo_root()
    events_path = (
        Path(events_path)
        if events_path is not None
        else repo_root / "datasets" / "events.jsonl"
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else repo_root / "datasets" / "tgn_edges_basic.npz"
    )

    if output_path.exists() and not force:
        print(
            f"[preprocess] {output_path} already exists — skipping (use force=True to rebuild)."
        )
        return output_path

    # Lazy load tokenizer/model to avoid importing heavy assets on module import
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    node2id: dict[str, int] = {}
    node_features: dict[int, np.ndarray] = {}
    src_nodes, dst_nodes, timestamps, edge_features = [], [], [], []

    # --- Utility functions for TGN preprocessing ---
    def get_node_id(node: str) -> int:
        """Get unique node ID for a given node"""
        if node not in node2id:
            node2id[node] = len(node2id)
        return node2id[node]

    def to_timestamp(ts: str) -> float:
        """Convert ISO8601 timestamp to float (seconds since epoch)"""
        from utils.datetime import parse_iso_timestamp

        dt = parse_iso_timestamp(ts)
        return dt.timestamp()

    def embed_text(text: str) -> np.ndarray:
        """Embed text using DistilBERT and return the mean pooling of the output."""
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_text_len
        )
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return emb

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    with events_path.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            user = get_node_id(event["user_id"])
            content = get_node_id(event["content_id"])
            time_val = to_timestamp(event["timestamp"])
            platform = event.get("source", "")
            e_type = event.get("type", "")

            # One-hot: [is_twitter, is_youtube, is_original, is_retweet, is_upload]
            feature = [
                int(platform == "twitter"),
                int(platform == "youtube"),
                int(e_type == "original"),
                int(e_type == "retweet"),
                int(e_type == "upload"),
            ]

            # Add edge: user — content
            src_nodes.append(user)
            dst_nodes.append(content)
            timestamps.append(time_val)
            edge_features.append(feature)

            # Extract content node embedding ONLY if first seen
            if content not in node_features:
                node_features[content] = embed_text(event.get("text", ""))

            # User node (NOTE: simple zeros for now)
            if user not in node_features:
                node_features[user] = np.zeros(768)

            # Hashtag nodes
            for hashtag in event.get("hashtags", []):
                h_id = get_node_id("h_" + hashtag)
                src_nodes.append(content)
                dst_nodes.append(h_id)
                timestamps.append(time_val)
                edge_features.append([0, 0, 0, 0, 0])  # TODO: Add meaningful features

    # --- Build full node features array (in node ID order) ---
    num_nodes = len(node2id)
    feature_dim = 768  # DistilBERT base output size
    features_list = []
    for i in range(num_nodes):
        if i in node_features:
            features_list.append(node_features[i])
        else:
            features_list.append(np.zeros(feature_dim))
    features_array = np.stack(features_list)

    # Save to .npz for easy loading in PyTorch
    np.savez(
        output_path,
        src=np.array(src_nodes),
        dst=np.array(dst_nodes),
        t=np.array(timestamps),
        edge_attr=np.array(edge_features),
        node_map=np.array(list(node2id.keys())),
        node_features=features_array,
    )

    print(
        f"[preprocess] Saved {len(src_nodes)} edges and {len(node2id)} nodes to {output_path}"
    )
    return output_path


if __name__ == "__main__":
    build_tgn()  # Allow running the script directly

