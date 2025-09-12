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
    GROWTH_HORIZON_H,
    LABEL_WINDOW_H,
    LABEL_TYPE,
    LABEL_EPS,
)
# ------------ Helpers ----------------
# exponential half-life per hour (consistent, simple)
def _recency_scalar(ts_sec: float, ref_sec: float) -> float:
    # half-life = 1 hour  →  rec = 0.5 ** hours
    hours = max(0.0, (ref_sec - ts_sec) / 3600.0)
    return float(0.5 ** hours)
    
def _bin_time(ts_sec: float, window_h: int) -> int:
    """Return an integer time bin index (hours // window_h)"""
    return int((ts_sec // 3600) // window_h)

def _series_to_future_labels(cur: float, fut: float, label_type: str, eps: float) -> float:
    if label_type == "diff":
        return float(fut - cur)
    elif label_type == "pct":
        return float((fut - cur) / max(cur, eps))
    elif label_type == "logdiff":
        return float(np.log(fut + eps) - np.log(cur + eps))
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    
def build_future_labels(events, k_hours: int, window_h: int, label_type: str, eps: float):
    """ 
    events: iterable of dicts with at least:
        - topic_id (int)  -> content node id
        - timestamp_sec (float)
        - weight (float)  -> engagement increment (default 1.0)
    Returns:
      labels: dict[(topic_id, time_bin)] -> float future growth label
      cur_counts, fut_counts (dicts) for optional diagnostics

    """
    from collections import defaultdict

    counts = defaultdict(float)  # (topic_id, time_bin) -> count
    for e in events:
        tb = _bin_time(e["timestamp_sec"], window_h)
        counts[(e["topic_id"], tb)] += float(e.get("weight", 1.0))
    k_bins = k_hours // window_h
    cur_counts, fut_counts, labels = {}, {}, {}

    # build labels
    for (topic_id, tb), cur in counts.items():
        fut_tb = tb + k_bins
        fut = counts.get((topic_id, fut_tb), 0.0)
        label = _series_to_future_labels(cur, fut, label_type, eps)
        cur_counts[(topic_id, tb)] = cur
        fut_counts[(topic_id, tb)] = fut
        labels[(topic_id, tb)] = label

    return labels, cur_counts, fut_counts



# -----------------------------------

def build_tgn(
    events_path: Path | str | None = None,
    output_path: Path | str | None = None,
    force: bool = False,
    max_text_len = EMBED_MAX_TOKENS,
) -> Optional[Path]:
    """Build TGN edge file from events.jsonl + future growth labels."""
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
    src_nodes, dst_nodes, timestamps = [], [], []
    edge_features = []  # will be rebuilt later

    edge_topic_ids: list[int] = []  # for future labels
    edge_time_bins: list[int] = []  # for future labels

    events_for_labels = []  # list of {"topic_id": int, "timestamp_sec": float, "weight": 1.0}

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
            # -------- Add edge: user — content --------
            src_nodes.append(user)
            dst_nodes.append(content)
            timestamps.append(time_val)
            # labels: topic is the content node, bin by config window
            edge_topic_ids.append(content)
            edge_time_bins.append(_bin_time(time_val, LABEL_WINDOW_H))

            # Count this as an engagement event for label building
            events_for_labels.append({
                "topic_id": content,
                "timestamp_sec": float(time_val),
                "weight": 1.0, # simple count; can later weight by platform/type
            })

            # ---- text embedding for content nodes ----
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
                # edge: content — hashtag
                src_nodes.append(content)
                dst_nodes.append(h_id)
                timestamps.append(time_val)
                # Label for this edge should still reference the content topic at this time
                edge_topic_ids.append(content)
                edge_time_bins.append(_bin_time(time_val, LABEL_WINDOW_H))
                # TODO: placeholders features are rebuilt later
                # (intentionally don't add hashtag edges to events_for_labels)

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

    # --- Build future growth labels ---
    labels_dict, cur_counts, fut_counts = build_future_labels(
        events=events_for_labels,
        k_hours=GROWTH_HORIZON_H,
        window_h=LABEL_WINDOW_H,
        label_type=LABEL_TYPE,
        eps=LABEL_EPS,
    )

    # Map per-edge
    y_growth = np.zeros(len(src_nodes), dtype=np.float32)
    for i, (topic, tb) in enumerate(zip(edge_topic_ids, edge_time_bins)):
        y_growth[i] = float(labels_dict.get((topic, tb), 0.0))

    y_growth = np.nan_to_num(y_growth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) # clean up


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
        ),
        # Predictive target exports
        y_growth=y_growth,
        edge_topic_ids=np.array(edge_topic_ids),
        edge_time_bins=np.array(edge_time_bins),
        growth_horizon_hours=np.array([GROWTH_HORIZON_H], dtype=np.int32),
        label_window_h=np.array([LABEL_WINDOW_H], dtype=np.int32),
        label_type=np.array([LABEL_TYPE]),
        label_eps=np.array([LABEL_EPS], dtype=np.float32),
    )

    
    print(
        f"[preprocess] Saved {len(src_nodes)} edges and {len(node2id)} nodes to {output_path}"
        f"(y_growth aligned to edges; horizon={GROWTH_HORIZON_H}h, window={LABEL_WINDOW_H}h, type={LABEL_TYPE})"
    )
    return output_path


if __name__ == "__main__":
    build_tgn()  # Allow running the script directly

