"""
Top-K predictions panel component for live trend monitoring.

Shows current Top-K predictions with countdown timers until Δ-freeze.
Use `render_panel()` to get the Streamlit component for Top-K live predictions.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st

from config.config import (
    PREDICTIONS_CACHE_PATH,
    DELTA_HOURS,
    K_DEFAULT,
    K_OPTIONS,
    TOPIC_LOOKUP_PATH,
    GROWTH_HORIZON_H,
)
from config.schemas import PredictionsCache, CacheItem


def _load_predictions_cache(cache_path: Path | str) -> Optional[PredictionsCache]:
    """Load predictions cache from JSON file."""
    try:
        cache_file = Path(cache_path)
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure matches PredictionsCache schema
        return data
    except Exception as e:
        st.error(f"Error loading predictions cache: {e}")
        return None


def _load_topic_lookup(path: Path | str) -> Dict[str, str]:
    """Load topic ID to label mapping."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _format_countdown(delta_hours: int, prediction_time: str) -> str:
    """Format countdown timer until Δ-freeze."""
    try:
        pred_dt = datetime.fromisoformat(prediction_time.replace('Z', '+00:00'))
        freeze_time = pred_dt + timedelta(hours=delta_hours)
        now = datetime.now(timezone.utc)
        
        if now >= freeze_time:
            return "⏰ Frozen"
        
        remaining = freeze_time - now
        hours = remaining.seconds // 3600
        minutes = (remaining.seconds % 3600) // 60
        
        if remaining.days > 0:
            return f"🕐 {remaining.days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"🕐 {hours}h {minutes}m"
        else:
            return f"🕐 {minutes}m"
            
    except Exception as e:
        return "⚠️ Invalid time"


def _get_latest_predictions(cache: PredictionsCache) -> Optional[CacheItem]:
    """Get the most recent prediction item from cache."""
    if not cache or not cache.get('items'):
        return None
    
    # Sort by timestamp to get latest
    items = cache['items']
    latest_item = None
    latest_time = None
    
    for item in items:
        try:
            item_time = datetime.fromisoformat(item['t_iso'].replace('Z', '+00:00'))
            if latest_time is None or item_time > latest_time:
                latest_time = item_time
                latest_item = item
        except Exception:
            continue
    
    return latest_item


def render_panel() -> Dict[str, Any]:
    """
    Render the Top-K live predictions panel.
    
    Returns:
        Dict containing panel state and metrics for external monitoring.
    """
    st.header("📈 Live Top-K Predictions")

    # Load predictions cache
    cache = _load_predictions_cache(PREDICTIONS_CACHE_PATH)
    topic_lookup = _load_topic_lookup(TOPIC_LOOKUP_PATH)
    
    if cache is None:
        st.warning("⚠️ No predictions cache found. Check if predictions are being generated.")
        return {"status": "no_cache", "predictions_count": 0}
    
    # Get latest predictions
    latest_predictions = _get_latest_predictions(cache)
    
    if latest_predictions is None:
        st.warning("📭 No predictions available in cache.")
        return {"status": "empty_cache", "predictions_count": 0}
    
    # K selector
    col1, col2 = st.columns([1, 3])
    k_options = list(K_OPTIONS)
    default_index = k_options.index(K_DEFAULT) if K_DEFAULT in k_options else 0
    with col1:
        selected_k = st.selectbox(
            "Top K:",
            options=list(K_OPTIONS),
            index=default_index,
            help=f"Show top K predictions (options: {K_OPTIONS})"
        )
    with col2:
        # Cache info
        last_updated = cache.get('last_updated', 'Unknown')
        try:
            updated_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            formatted_time = updated_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            formatted_time = last_updated
        
        st.info(f"📅 Cache last updated: {formatted_time}")
    
    # Display predictions
    all_predictions = latest_predictions.get('topk', [])
    total_available = len(all_predictions)
    topk_predictions = all_predictions[:selected_k]
    prediction_time = latest_predictions['t_iso']
    
    if not topk_predictions:
        st.warning("📭 No predictions available for the selected K value.")
        return {"status": "no_predictions", "predictions_count": 0}
    
    st.subheader(f"🎯 Top {selected_k} Predictions")
    
    # Create columns for the prediction table
    col_rank, col_topic, col_score, col_countdown = st.columns([1, 2, 2, 3])
    
    with col_rank:
        st.write("**Rank**")
    with col_topic:
        st.write("**Topic**")
    with col_score:
        st.write(f"**Predicted Growth (next {GROWTH_HORIZON_H}h)**")
    with col_countdown:
        st.write("**Time to Δ-freeze**")
    
    # Display each prediction
    for i, pred in enumerate(topk_predictions, 1):
        topic_id = pred.get('topic_id', 'Unknown')
        score = pred.get('score', 0.0)
        countdown = _format_countdown(DELTA_HOURS, prediction_time)

        col_rank, col_topic, col_score, col_countdown = st.columns([1, 2, 2, 3])

        with col_rank:
            st.write(f"#{i}")
        with col_topic:
            label = topic_lookup.get(str(topic_id))
            if label:
                st.write(f"{label} (`{topic_id}`)")
            else:
                st.write(f"`{topic_id}`")
        with col_score:
            st.metric("", f"{score:.3f}")
        with col_countdown:
            st.write(countdown)
    
    # Additional info
    st.divider()
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.metric("🎯 Total Predictions", len(topk_predictions))
        
    with col_info2:
        st.metric("⏱️ Δ-freeze Window", f"{DELTA_HOURS}h")
    
    # Configuration info (for debugging/transparency)
    with st.expander("🔧 Configuration"):
        st.write(f"**Cache Path:** `{PREDICTIONS_CACHE_PATH}`")
        st.write(f"**K Options:** `{K_OPTIONS}`")
        st.write(f"**Delta Hours:** `{DELTA_HOURS}`")
        st.write(f"**Prediction Time:** `{prediction_time}`")
    
    return {
        "status": "success",
        "predictions_count": len(topk_predictions),
        "available_count": total_available,
        "selected_k": selected_k,
        "prediction_time": prediction_time,
        "cache_last_updated": last_updated
    }