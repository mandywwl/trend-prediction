import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from transformers import DistilBertTokenizer, DistilBertModel

from config.config import (
    DATA_DIR,
    TGN_EDGE_DIM,
    EMBED_MAX_TOKENS,
    EDGE_WEIGHT_MIN,
)
# ------------ Helpers ----------------
# exponential half-life per hour (consistent, simple)
def _recency_scalar(ts_sec: float, ref_sec: float) -> float:
    # half-life = 1 hour  →  rec = 0.5 ** hours
    hours = max(0.0, (ref_sec - ts_sec) / 3600.0)
    return float(0.5 ** hours)
    
# -----------------------------------

def build_tgn(
    events_path: Path | str | None = None,
    output_path: Path | str | None = None,
    force: bool = False,
    max_text_len = EMBED_MAX_TOKENS,
) -> Optional[Path]:
    """Build TGN edge file from events.jsonl."""
    events_path = (
        Path(events_path)
        if events_path is not None
        else DATA_DIR / "events.jsonl"
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else DATA_DIR / "tgn_edges_basic.npz"
    )

    if output_path.exists() and not force:
        print(
            f"[preprocess] {output_path} already exists — skipping (use force=True to rebuild)."
        )
        return output_path

    # Lazy load tokenizer/model to avoid importing heavy assets on module import
    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        model.eval()  # inference mode
    except Exception as e:
        print(f"[preprocess] Error loading DistilBERT model: {e}")
        return None

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
        from utils.datetime import parse_iso_timestamp # reuse existing utility

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
    # ---------------------------------------------

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    with events_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)

            # ---- normalize core IDs ----
            user_raw = (
                event.get("user_id")
                or event.get("actor_id")
                or event.get("author_id")
                or event.get("u_id")
                or event.get("channel_id")
                or event.get("account_id")
                or "unknown_user"
            )
            content_raw = (
                event.get("content_id")
                or event.get("video_id")
                or event.get("tweet_id")
                or event.get("post_id")
                or (event.get("target_id"))
                or (event.get("target_ids", [None])[0] if isinstance(event.get("target_ids"), list) and event.get("target_ids") else None)
                or event.get("id")
                or event.get("text")  # last resort: use text as content key
                or f"content_{len(node2id)}"
            )
            ts_raw = (
                event.get("timestamp")
                or event.get("ts_iso")
                or event.get("created_at")
                or datetime.now(timezone.utc).isoformat() # fallback
            )
            user = get_node_id(str(user_raw))
            content = get_node_id(str(content_raw))
            time_val = to_timestamp(ts_raw)

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

            # ---- text embedding for content nodes ----

            # prefer common fields
            text = (
                event.get("text")
                or event.get("tweet_text")
                or event.get("caption")
                or event.get("title")
                or event.get("description")
                or ""
            )
            if content not in node_features:
                try:
                    node_features[content] = embed_text(text)
                except Exception:
                    node_features[content] = np.zeros(TGN_EDGE_DIM, dtype=np.float32)

            # User node (NOTE: simple zeros for now)
            if user not in node_features:
                node_features[user] = np.zeros(TGN_EDGE_DIM, dtype=np.float32)

            # ---------- Hashtag/tags edges ----------
            tags = event.get("hashtags") or event.get("tags") or []
            if isinstance(tags, str):
                # naive split for comma/space-separated strings
                parts = [t for t in tags.replace(",", " ").split() if t]
            elif isinstance(tags, list):
                parts = [str(t) for t in tags if t]
            else:
                parts = []


            for hashtag in parts:
                h_id = get_node_id("h_" + hashtag)
                src_nodes.append(content)
                dst_nodes.append(h_id)
                timestamps.append(time_val)
                edge_features.append([0, 0, 0, 0, 0])  # TODO: Add richer features later

    # --- Build full node features array (in node ID order) ---
    num_nodes = len(node2id)
    features_list = []
    for i in range(num_nodes):
        if i in node_features:
            features_list.append(node_features[i])
        else:
            features_list.append(np.zeros(TGN_EDGE_DIM, dtype=np.float32))
    features_array = np.stack(features_list)

    # --- Finalize edge features with recency ---
    # reference time: use the latest timestamp in this dataset
    ref_sec = float(max(timestamps)) if timestamps else float(datetime.now(timezone.utc).timestamp())

    edge_features = []  # rebuild as [768 emb ..., 1 recency]
    for s, d, t in zip(src_nodes, dst_nodes, timestamps):
        # content/node embedding for this edge (zeros if missing)
        emb = node_features.get(d)
        if emb is None:
            emb = np.zeros(TGN_EDGE_DIM, dtype=np.float32)

        # recency scalar in [0,1]
        rec = _recency_scalar(float(t), ref_sec)
        rec = max(EDGE_WEIGHT_MIN, rec)  # clamp like runtime

        # concat → [768 + 1]
        vec = np.concatenate([emb.astype(np.float32, copy=False),
                            np.array([rec], dtype=np.float32)], axis=0)
        edge_features.append(vec)

    edge_features = np.stack(edge_features, axis=0)

    # Save to .npz for easy loading in PyTorch
    np.savez(
        output_path,
        src=np.array(src_nodes),
        dst=np.array(dst_nodes),
        t=np.array(timestamps),
        edge_attr=edge_features,
        node_map=np.array(list(node2id.keys())),
        node_features=np.stack(
            [node_features.get(i, np.zeros(TGN_EDGE_DIM, dtype=np.float32)) for i in range(len(node2id))]
        )
    )

    print(
        f"[preprocess] Saved {len(src_nodes)} edges and {len(node2id)} nodes to {output_path}"
    )
    return output_path


if __name__ == "__main__":
    build_tgn()  # Allow running the script directly

