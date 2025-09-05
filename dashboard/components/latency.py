"""
Latency panel component for monitoring SLOs and latency trends.

Visualizes per-hour median/p95 latency metrics and surfaces SLO breaches.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from config.config import SLO_MED_MS, SLO_P95_MS, METRICS_SNAPSHOT_DIR
from config.schemas import HourlyMetrics, LatencySummary, StageMs
from components.layout import (
    render_metric_card, 
    render_stage_table, 
    render_breach_indicator,
    apply_custom_css
)


def _load_hourly_metrics(metrics_dir: str) -> List[Tuple[datetime, HourlyMetrics]]:
    """
    Load hourly metrics files from the metrics directory.
    
    Args:
        metrics_dir: Path to metrics directory
        
    Returns:
        List of (timestamp, metrics) tuples sorted by timestamp
    """
    metrics_path = Path(metrics_dir)
    if not metrics_path.exists():
        return []
    
    metrics_data = []
    
    # Look for metrics files matching pattern metrics_YYYYMMDD_HH.json
    for file_path in metrics_path.glob("metrics_*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp from filename
            filename = file_path.stem  # metrics_YYYYMMDD_HH
            date_str = filename.replace('metrics_', '')
            
            # Parse timestamp
            if len(date_str) == 11 and '_' in date_str:  # YYYYMMDD_HH
                date_part, hour_part = date_str.split('_')
                year = int(date_part[:4])
                month = int(date_part[4:6])
                day = int(date_part[6:8])
                hour = int(hour_part)
                
                timestamp = datetime(year, month, day, hour, tzinfo=timezone.utc)
                metrics_data.append((timestamp, data))
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Skip invalid files silently
            continue
    
    # Sort by timestamp
    metrics_data.sort(key=lambda x: x[0])
    return metrics_data


def _safe_parse_latency(data: Dict[str, Any]) -> Optional[LatencySummary]:
    """
    Safely parse latency data from hourly metrics.
    
    Args:
        data: Raw metrics data dictionary
        
    Returns:
        LatencySummary if valid, None otherwise
    """
    try:
        latency_data = data.get('latency', {})
        if not latency_data:
            return None
        
        # Extract per-stage data safely
        per_stage_data = latency_data.get('per_stage_ms', {})
        per_stage_ms = StageMs(
            ingest=int(per_stage_data.get('ingest', 0)),
            preprocess=int(per_stage_data.get('preprocess', 0)),
            model_update_forward=int(per_stage_data.get('model_update_forward', 0)),
            postprocess=int(per_stage_data.get('postprocess', 0))
        )
        
        return LatencySummary(
            median_ms=int(latency_data.get('median_ms', 0)),
            p95_ms=int(latency_data.get('p95_ms', 0)),
            per_stage_ms=per_stage_ms
        )
        
    except (ValueError, TypeError, KeyError):
        return None


def _filter_slo_breaches(
    metrics_data: List[Tuple[datetime, HourlyMetrics]],
    show_breaches_only: bool
) -> List[Tuple[datetime, LatencySummary]]:
    """
    Filter metrics data and extract latency summaries.
    
    Args:
        metrics_data: Raw metrics data
        show_breaches_only: If True, only return entries with SLO breaches
        
    Returns:
        Filtered list of (timestamp, latency_summary) tuples
    """
    filtered_data = []
    
    for timestamp, raw_metrics in metrics_data:
        latency = _safe_parse_latency(raw_metrics)
        if not latency:
            continue
        
        # Check if this entry has SLO breaches
        has_breach = (
            latency['median_ms'] >= SLO_MED_MS or 
            latency['p95_ms'] >= SLO_P95_MS
        )
        
        # Include if not filtering, or if filtering and has breach
        if not show_breaches_only or has_breach:
            filtered_data.append((timestamp, latency))
    
    return filtered_data


def _render_latency_chart(data: List[Tuple[datetime, LatencySummary]]) -> plt.Figure:
    """
    Create a line chart showing median and p95 latency trends.
    
    Args:
        data: List of (timestamp, latency_summary) tuples
        
    Returns:
        Matplotlib figure
    """
    if not data:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No latency data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Latency Trends")
        return fig
    
    # Extract data for plotting
    timestamps = [item[0] for item in data]
    median_values = [item[1]['median_ms'] for item in data]
    p95_values = [item[1]['p95_ms'] for item in data]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot lines
    ax.plot(timestamps, median_values, label='Median', color='#1f77b4', linewidth=2)
    ax.plot(timestamps, p95_values, label='P95', color='#ff7f0e', linewidth=2)
    
    # Add SLO threshold lines
    ax.axhline(y=SLO_MED_MS, color='#1f77b4', linestyle='--', alpha=0.7, 
               label=f'Median SLO ({SLO_MED_MS}ms)')
    ax.axhline(y=SLO_P95_MS, color='#ff7f0e', linestyle='--', alpha=0.7,
               label=f'P95 SLO ({SLO_P95_MS}ms)')
    
    # Formatting
    ax.set_title("Latency Trends (Last 24h)")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    if len(timestamps) > 0:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig


def render_panel(
    *,
    metrics_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Render the latency monitoring panel.
    
    Args:
        metrics_dir: Path to metrics directory (uses config default if None)
        
    Returns:
        Dictionary with panel status and metrics for external monitoring
    """
    # Apply custom CSS
    apply_custom_css()
    
    # Header
    st.header("⏱️ Latency & SLOs")
    
    # Use default metrics directory if not provided
    if metrics_dir is None:
        metrics_dir = METRICS_SNAPSHOT_DIR
    
    # Load metrics data
    metrics_data = _load_hourly_metrics(metrics_dir)
    
    if not metrics_data:
        st.warning("⚠️ No latency metrics found. Check if metrics are being collected.")
        return {
            "status": "no_data",
            "breach_count": 0,
            "total_count": 0
        }
    
    # Controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Current Status")
    
    with col2:
        show_breaches_only = st.checkbox(
            "🚨 Show breaches only",
            value=False,
            help="Filter to show only hours with SLO breaches"
        )
    
    # Filter data
    filtered_data = _filter_slo_breaches(metrics_data, show_breaches_only)
    
    if not filtered_data:
        if show_breaches_only:
            st.success("🎉 No SLO breaches found in the data!")
        else:
            st.warning("⚠️ No valid latency data available.")
        return {
            "status": "no_filtered_data",
            "breach_count": 0,
            "total_count": len(metrics_data)
        }
    
    # Get latest metrics for current status
    latest_timestamp, latest_latency = filtered_data[-1]
    
    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            label="Median Latency",
            value=latest_latency['median_ms'],
            threshold=SLO_MED_MS,
            unit="ms"
        )
    
    with col2:
        render_metric_card(
            label="P95 Latency", 
            value=latest_latency['p95_ms'],
            threshold=SLO_P95_MS,
            unit="ms"
        )
    
    with col3:
        render_metric_card(
            label="SLO Med Threshold",
            value=SLO_MED_MS,
            unit="ms"
        )
    
    with col4:
        render_metric_card(
            label="SLO P95 Threshold",
            value=SLO_P95_MS,
            unit="ms"
        )
    
    # SLO Breach Status
    median_breach = latest_latency['median_ms'] >= SLO_MED_MS
    p95_breach = latest_latency['p95_ms'] >= SLO_P95_MS
    
    render_breach_indicator(median_breach, p95_breach, SLO_MED_MS, SLO_P95_MS)
    
    # Latency Trend Chart
    st.subheader("📊 Latency Trends")
    
    fig = _render_latency_chart(filtered_data)
    st.pyplot(fig)
    
    # Per-stage Breakdown  
    st.subheader("🔧 Per-Stage Breakdown")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Latest Hour Breakdown:**")
        render_stage_table(latest_latency['per_stage_ms'])
    
    with col2:
        # Calculate breach statistics
        total_hours = len(filtered_data)
        breach_hours = sum(
            1 for _, latency in filtered_data
            if latency['median_ms'] >= SLO_MED_MS or latency['p95_ms'] >= SLO_P95_MS
        )
        
        st.write("**SLO Breach Summary:**")
        st.metric(
            label="Breach Hours",
            value=f"{breach_hours}/{total_hours}",
            delta=f"{(breach_hours/total_hours*100):.1f}%" if total_hours > 0 else "0%"
        )
        
        st.write(f"📅 **Data Range:** {len(filtered_data)} hours")
        if filtered_data:
            oldest_time = filtered_data[0][0]
            newest_time = filtered_data[-1][0]
            st.write(f"From {oldest_time.strftime('%Y-%m-%d %H:%M')} to {newest_time.strftime('%Y-%m-%d %H:%M')} UTC")
    
    return {
        "status": "success",
        "breach_count": breach_hours,
        "total_count": total_hours,
        "latest_median_ms": latest_latency['median_ms'],
        "latest_p95_ms": latest_latency['p95_ms'],
        "median_slo_met": not median_breach,
        "p95_slo_met": not p95_breach
    }