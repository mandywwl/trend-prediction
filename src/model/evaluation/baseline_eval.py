from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict, Counter
import math
import csv
import pandas as pd
import numpy as np
from service.tgn_service import TGNInferenceService

try:
    from config.config import (
        DATA_DIR,
        WINDOW_MINUTES, 
        HORIZON_MINUTES, 
        EMERGENCE_THRESHOLD,
        MIN_GROWTH, 
        MIN_SUPPORT, 
        K_VALUES, 
        HOLDOUT_DAYS
    )
    CFG = dict(
        WINDOW_MINUTES=WINDOW_MINUTES,
        HORIZON_MINUTES=HORIZON_MINUTES,
        EMERGENCE_THRESHOLD=EMERGENCE_THRESHOLD,
        MIN_GROWTH=MIN_GROWTH,
        MIN_SUPPORT=MIN_SUPPORT,
        K_VALUES=K_VALUES,
        HOLDOUT_DAYS=HOLDOUT_DAYS,
    )
except ImportError:
    # Default config if not provided
    CFG = dict(
        WINDOW_MINUTES=60,
        HORIZON_MINUTES=60,
        EMERGENCE_THRESHOLD=5,
        MIN_GROWTH=3,
        MIN_SUPPORT=1,
        K_VALUES=[1, 5, 10, 20],
        HOLDOUT_DAYS=0,
    )
    print("Using default config values. To customize, edit config/config.py with desired settings.")

@dataclass
class WindowSpec:
    window_minutes: int
    horizon_minutes: int

# ----------------- Data loading helpers -----------------
def load_events(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Flatten {"event": {...}} if present
                if isinstance(rec.get("event"), dict):
                    rec = {**rec, **rec["event"]}
                rows.append(rec)
        df = pd.json_normalize(rows, sep=".")
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported events file type: {p.suffix}")

    # ---- coalesce item_id ----
    # Try common direct IDs (INCLUDE content_id here)
    id_candidates = [
        "item_id",
        "content_id", "content.id",
        "hashtag", "topic", "term", "keyword", "tag", "name", "label",
        "tweet_id", "tweet.id", "id",
        "video_id", "videoId", "youtube.video_id",
        "target_id",
    ]
    src = next((c for c in id_candidates if c in df.columns), None)

    # Derive from target_ids if present (use first)
    if src is None and "target_ids" in df.columns:
        df["item_id"] = df["target_ids"].apply(
            lambda v: (v[0] if isinstance(v, (list, tuple)) and v else None)
        )
        src = "item_id"

    # Derive from hashtags (use first) if still not found
    if src is None and "hashtags" in df.columns:
        df["item_id"] = df["hashtags"].apply(
            lambda hs: (hs[0] if isinstance(hs, (list, tuple)) and hs else None)
        )
        src = "item_id"

    if src is None:
        raise ValueError(
            "events must include an identifier (one of: item_id/content_id/hashtag/topic/term/keyword/tag/name/"
            "label/tweet_id/video_id/id/target_ids/hashtags)"
        )

    if src != "item_id":
        df["item_id"] = df[src]

    # Normalise simple hashtags (strip leading '#'), lowercase
    df["item_id"] = df["item_id"].astype(str).str.strip().str.lstrip("#").str.lower()

    # ---- coalesce timestamp ----
    ts_candidates = [
        "ts", "ts_iso", "timestamp", "created_at", "time",
        "publishedAt", "snippet.publishedAt", "createdAt",
        "ts_s",  # already-in-seconds case
    ]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            "events must include a timestamp column (one of: ts/ts_iso/timestamp/created_at/time/publishedAt/ts_s)"
        )

    if ts_col == "ts_s":
        df["ts_s"] = pd.to_numeric(df["ts_s"], errors="coerce").astype("Int64")
        if df["ts_s"].isna().any():
            raise ValueError("Unparsable numeric timestamps in 'ts_s'")
        df["ts"] = pd.to_datetime(df["ts_s"].astype(np.int64), unit="s", utc=True)
    else:
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if df["ts"].isna().any():
            bad = df[df["ts"].isna()].head(3)
            raise ValueError(
                f"Found unparsable timestamps in '{ts_col}', examples:\n"
                f"{bad[[ts_col]].to_dict(orient='records')}"
            )
        # pandas 2.x friendly
        df["ts_s"] = (df["ts"].astype("int64") // 10**9).astype(np.int64)

    # Optional weights
    if "weight" not in df.columns:
        df["weight"] = 1.0

    # ---- optional: explode hashtags as additional items ----
    INCLUDE_HASHTAGS = True
    if INCLUDE_HASHTAGS and "hashtags" in df.columns:
        # Guard against non-iterables/NaNs
        mask = df["hashtags"].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)
        if mask.any():
            tag_df = df.loc[mask].copy()
            tag_df = tag_df.explode("hashtags")
            tag_df["item_id"] = (
                tag_df["hashtags"].astype(str).str.strip().str.lstrip("#").str.lower()
            )
            tag_df = tag_df.drop(columns=["hashtags"])
            df = pd.concat([df, tag_df], ignore_index=True)

    return df[["item_id", "ts", "ts_s", "weight"]].sort_values("ts")

# ----------------- Windowing and ground truth -----------------
def generate_cutpoints(df: pd.DataFrame, window_minutes: int, horizon_minutes: int, holdout_days: int = 0) -> List[int]:
    """Return list of window_end timestamps (in seconds) where we will evaluate."""
    tmin = int(df["ts_s"].min())
    tmax = int(df["ts_s"].max())
    step = window_minutes * 60
    horizon = horizon_minutes * 60

    if holdout_days and holdout_days > 0:
        tmax_eval = tmax - holdout_days * 86400
    else:
        tmax_eval = tmax

    # ensure we have horizon future room
    tmax_eval = tmax_eval - horizon
    if tmax_eval <= tmin + step:
        return []
    cutpoints = list(range(tmin + step, tmax_eval + 1, step))
    return cutpoints

def slice_counts(df: pd.DataFrame, start_s: int, end_s: int) -> Counter:
    """Weighted counts per item_id in [start_s, end_s]."""
    m = (df["ts_s"] >= start_s) & (df["ts_s"] <= end_s)
    if not m.any(): return Counter()
    g = df.loc[m].groupby("item_id")["weight"].sum()
    return Counter(g.to_dict())

def build_ground_truth(
    df: pd.DataFrame,
    t_end: int,
    window_minutes: int,
    horizon_minutes: int,
    emergence_threshold: int,
    min_growth: int,
    min_support: int,
) -> Tuple[set, Dict[str, Dict[str, float]]]:
    """
    Return (emerged_items, diagnostics) where emerged_items are item_ids that
    were not "hot" before t_end but become hot (or grow) in (t_end, t_end+horizon].
    """
    window_s = window_minutes * 60
    horizon_s = horizon_minutes * 60

    past_start = t_end - window_s + 1
    future_start = t_end + 1
    future_end = t_end + horizon_s

    past = slice_counts(df, past_start, t_end)
    future = slice_counts(df, future_start, future_end)

    # Candidate set = items seen in past (support) ∪ newly appearing in future
    candidates = set(past.keys()).union(future.keys())

    emerged = set()
    diag: Dict[str, Dict[str, float]] = {}
    for item in candidates:
        past_c = past.get(item, 0.0)
        fut_c = future.get(item, 0.0)
        growth = fut_c - past_c
        # "Not already hot" = below threshold at t_end and min support satisfied (or zero)
        not_hot = past_c < emergence_threshold
        support_ok = (past_c >= min_support) or (past_c == 0)
        became_hot = fut_c >= emergence_threshold
        grew_enough = growth >= min_growth
        if not_hot and support_ok and (became_hot or grew_enough):
            emerged.add(item)
        diag[item] = {"past": past_c, "future": fut_c, "growth": growth}
    return emerged, diag

# ----------------- Baselines -----------------
def baseline_popularity(past_counts: Counter) -> Dict[str, float]:
    # More popular in past window -> higher score (predicts “will continue”)
    # For emergence, this is intentionally a weak baseline.
    return dict(past_counts)

def baseline_momentum(past_counts: Counter, prev_counts: Counter) -> Dict[str, float]:
    # Simple momentum: difference between current window and last window
    scores = {}
    items = set(past_counts.keys()).union(prev_counts.keys())
    for it in items:
        scores[it] = past_counts.get(it, 0.0) - prev_counts.get(it, 0.0)
    return scores

def baseline_recency(df: pd.DataFrame, t_end: int, window_minutes: int) -> Dict[str, float]:
    # Higher score for items whose last occurrence is closer to t_end
    window_s = window_minutes * 60
    start = t_end - window_s + 1
    m = (df["ts_s"] >= start) & (df["ts_s"] <= t_end)
    if not m.any(): return {}
    tmp = df.loc[m].groupby("item_id")["ts_s"].max()
    # score = inverse recency gap
    return {k: 1.0 / max(1, (t_end - int(v))) for k, v in tmp.to_dict().items()}


# ----------------- Metrics -----------------
def precision_at_k(y_true: set, ranked_items: List[str], k: int) -> float:
    if k <= 0: return 0.0
    topk = ranked_items[:k]
    hits = sum(1 for it in topk if it in y_true)
    return hits / k

def hitrate_at_k(y_true: set, ranked_items: List[str], k: int) -> float:
    topk = set(ranked_items[:k])
    return 1.0 if len(y_true & topk) > 0 else 0.0

def ndcg_at_k(y_true: set, ranked_items: List[str], k: int) -> float:
    dcg = 0.0
    for i, it in enumerate(ranked_items[:k], start=1):
        rel = 1.0 if it in y_true else 0.0
        if rel > 0:
            dcg += 1.0 / math.log2(i + 1)
    # ideal DCG with |y_true| relevant items
    ideal_rels = [1.0] * min(k, len(y_true))
    idcg = sum(1.0 / math.log2(i + 1) for i, _ in enumerate(ideal_rels, start=1))
    return 0.0 if idcg == 0 else dcg / idcg

# ----------------- TGN -----------------
_SVC = None
def _svc() -> TGNInferenceService:
    global _SVC
    if _SVC is None:
        _SVC = TGNInferenceService(checkpoint_path= DATA_DIR / "tgn_model.pt")
    return _SVC

def _row_to_event(row) -> dict:
    """
    Convert a raw events row into the Event schema the service expects.
    Minimal fields are fine; if you don't have actor ids, reuse the item_id.
    """
    ts_iso = pd.to_datetime(row["ts"], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    item = str(row["item_id"])
    actor = str(row.get("actor_id", item))  # fallback if actor missing
    features = {}
    # If you carry text_emb or edge_weight on the row, pass them through:
    if "text_emb" in row:
        features["text_emb"] = row["text_emb"]
    if "edge_weight" in row:
        features["edge_weight"] = row["edge_weight"]
    return {
        "event_id": f"ev-{item}-{row['ts_s']}",
        "ts_iso": ts_iso,
        "actor_id": actor,
        "target_ids": [item],
        "edge_type": "observed",
        "features": features or None,
    } 
# ----------------- Main evaluation loop -----------------
def evaluate(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ws = WindowSpec(CFG["WINDOW_MINUTES"], CFG["HORIZON_MINUTES"])
    cutpoints = generate_cutpoints(df, ws.window_minutes, ws.horizon_minutes, CFG["HOLDOUT_DAYS"])
    if not cutpoints:
        raise RuntimeError("No evaluation cutpoints available. Check time coverage / horizon / holdout.")
    
    # Sort once, then stream events forward while advancing cutpnts
    df = df.sort_values("ts_s").reset_index(drop=True)
    i = 0  # index into df
    n = len(df)
    service = _svc()

    # Precompute previous window counts for momentum baseline
    prev_cache: Dict[int, Counter] = {}
    window_s = ws.window_minutes * 60
    for t_end in cutpoints:
        prev_start = t_end - 2*window_s + 1
        prev_end = t_end - window_s
        prev_cache[t_end] = slice_counts(df, prev_start, prev_end)

    per_window_rows = []
    for idx, t_end in enumerate(cutpoints):

        # Feed all events up to t_end into TGN memory
        while i < n and int(df.at[i, "ts_s"]) <= t_end:
            ev = _row_to_event(df.iloc[i])
            service.update_and_score(ev)  # ignore returned score here
            i += 1

        past_start = t_end - window_s + 1
        past_counts = slice_counts(df, past_start, t_end)

        # Candidates = items with any past support
        candidate_items = [it for it, c in past_counts.items() if c >= CFG["MIN_SUPPORT"]]

        # Build ground truth from future window
        y_true, diag = build_ground_truth(
            df=df,
            t_end=t_end,
            window_minutes=ws.window_minutes,
            horizon_minutes=ws.horizon_minutes,
            emergence_threshold=CFG["EMERGENCE_THRESHOLD"],
            min_growth=CFG["MIN_GROWTH"],
            min_support=CFG["MIN_SUPPORT"],
        )

        # Baselines
        scores_pop = baseline_popularity(past_counts)
        scores_mom = baseline_momentum(past_counts, prev_cache[t_end])
        scores_rec = baseline_recency(df, t_end, ws.window_minutes)

        # TGN score at t_end without mutating memory
        scores_tgn = service.score_items_at_time(candidate_items, t_end, mutate=False)
        
        # Rank each system
        def rank(scores: Dict[str, float]) -> List[str]:
            # descending by score, stable
            return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

        ranks = {
            "tgn": rank(scores_tgn),
            "popularity": rank(scores_pop),
            "momentum": rank(scores_mom),
            "recency": rank(scores_rec),
        }

        # Metrics
        for name, ranked in ranks.items():
            for k in CFG["K_VALUES"]:
                row = {
                    "t_end": t_end,
                    "system": name,
                    "k": k,
                    "precision_at_k": precision_at_k(y_true, ranked, k),
                    "hitrate_at_k": hitrate_at_k(y_true, ranked, k),
                    "ndcg_at_k": ndcg_at_k(y_true, ranked, k),
                    "num_emerged": len(y_true),
                    "num_candidates": len(ranked),
                }
                per_window_rows.append(row)

        # Optional: write diagnostics per window (small JSON)
        with (outdir / f"diag_{t_end}.json").open("w", encoding="utf-8") as f:
            json.dump({"t_end": t_end, "ground_truth_size": len(y_true)}, f)

    # Save per-window metrics
    dfw = pd.DataFrame(per_window_rows)
    dfw.to_csv(outdir / "per_window_metrics.csv", index=False)

    # Aggregate
    agg = (
        dfw.groupby(["system", "k"])
        .agg(
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            hitrate_at_k_mean=("hitrate_at_k", "mean"),
            ndcg_at_k_mean=("ndcg_at_k", "mean"),
            windows=("precision_at_k", "count"),
        )
        .reset_index()
        .sort_values(["k", "precision_at_k_mean"], ascending=[True, False])
    )
    agg.to_csv(outdir / "aggregate_metrics.csv", index=False)
    print("\n=== Aggregate (predictive) metrics ===")
    print(agg.to_string(index=False))

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser("Predictive baseline/TGN evaluation")
    ap.add_argument("--events", required=True, help="Path to events file (jsonl/csv/parquet)")
    ap.add_argument("--outdir", default="data/eval_predictive", help="Output directory for metrics")
    # Optional overrides
    ap.add_argument("--window-minutes", type=int)
    ap.add_argument("--horizon-minutes", type=int)
    ap.add_argument("--emergence-threshold", type=int)
    ap.add_argument("--min-growth", type=int)
    ap.add_argument("--min-support", type=int)
    ap.add_argument("--k", type=int, nargs="+")
    ap.add_argument("--holdout-days", type=int)
    return ap.parse_args()

def main():
    args = parse_args()
    for k, v in [
        ("WINDOW_MINUTES", args.window_minutes),
        ("HORIZON_MINUTES", args.horizon_minutes),
        ("EMERGENCE_THRESHOLD", args.emergence_threshold),
        ("MIN_GROWTH", args.min_growth),
        ("MIN_SUPPORT", args.min_support),
        ("K_VALUES", args.k),
        ("HOLDOUT_DAYS", args.holdout_days),
    ]:
        if v is not None:
            CFG[k] = v

    outdir = Path(args.outdir)
    df = load_events(args.events)
    evaluate(df, outdir)

if __name__ == "__main__":
    main()
