"""
Shared layout helpers for dashboard components.

Provides common UI utilities for colors, formatting, and layout consistency.
"""

from typing import Any, Dict, Optional
import streamlit as st


def slo_color(value: int, threshold: int, invert: bool = False) -> str:
    """
    Get color for SLO-based display.
    
    Args:
        value: Current metric value
        threshold: SLO threshold value
        invert: If True, green when value > threshold (for throughput metrics)
                If False, green when value < threshold (for latency metrics)
    
    Returns:
        Color string for CSS/Streamlit styling
    """
    if invert:
        return "green" if value > threshold else "red"
    else:
        return "green" if value < threshold else "red"


def format_latency(ms: int) -> str:
    """
    Format latency value for display.
    
    Args:
        ms: Latency in milliseconds
        
    Returns:
        Formatted string with appropriate unit
    """
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    else:
        return f"{ms}ms"


def render_metric_card(
    label: str, 
    value: int, 
    threshold: Optional[int] = None,
    unit: str = "ms",
    delta: Optional[int] = None
) -> None:
    """
    Render a metric card with SLO-aware coloring.
    
    Args:
        label: Metric label/title
        value: Current metric value
        threshold: SLO threshold for coloring (optional)
        unit: Unit string to display
        delta: Change from previous value (optional)
    """
    formatted_value = f"{value}{unit}"
    
    if threshold is not None:
        color = slo_color(value, threshold)
        if color == "red":
            formatted_value = f"🔴 {formatted_value}"
        else:
            formatted_value = f"🟢 {formatted_value}"
    
    st.metric(
        label=label,
        value=formatted_value,
        delta=f"{delta:+d}{unit}" if delta is not None else None
    )


def render_stage_table(per_stage_ms: Dict[str, int]) -> None:
    """
    Render per-stage latency breakdown table.
    
    Args:
        per_stage_ms: Dictionary mapping stage names to latency values
    """
    # Create a clean table display
    table_data = []
    for stage, ms in per_stage_ms.items():
        # Format stage name for display
        display_name = stage.replace("_", " ").title()
        table_data.append({
            "Stage": display_name,
            "Latency": format_latency(ms)
        })
    
    st.table(table_data)


def render_breach_indicator(
    median_breach: bool,
    p95_breach: bool,
    median_threshold: int,
    p95_threshold: int
) -> None:
    """
    Render SLO breach status indicator.
    
    Args:
        median_breach: True if median SLO is breached
        p95_breach: True if p95 SLO is breached  
        median_threshold: Median SLO threshold
        p95_threshold: P95 SLO threshold
    """
    if median_breach or p95_breach:
        st.error("🚨 **SLO Breach Alert**")
        
        if median_breach:
            st.write(f"• Median latency exceeds {median_threshold}ms SLO")
        if p95_breach:
            st.write(f"• P95 latency exceeds {p95_threshold}ms SLO")
    else:
        st.success("✅ **All SLOs Met**")
        st.write(f"• Median < {median_threshold}ms, P95 < {p95_threshold}ms")


def apply_custom_css() -> None:
    """Apply custom CSS styling for compact layout."""
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    
    .metric-good {
        border-left: 4px solid #28a745;
    }
    
    .metric-bad {
        border-left: 4px solid #dc3545;
    }
    
    .stage-table {
        font-size: 0.9rem;
    }
    
    .breach-alert {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)