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
    
    # Determine colors based on threshold
    bg_color = "#ffffff"
    text_color = "#212529"
    border_color = "#dee2e6"
    
    if threshold is not None:
        color = slo_color(value, threshold)
        if color == "red":
            formatted_value = f"🔴 {formatted_value}"
            bg_color = "#f8d7da"
            text_color = "#721c24"
            border_color = "#dc3545"
        else:
            formatted_value = f"🟢 {formatted_value}"
            bg_color = "#d4edda"
            text_color = "#155724"
            border_color = "#28a745"
    
    # Create custom HTML metric card with inline styles for guaranteed visibility
    delta_html = ""
    if delta is not None:
        delta_color = "#28a745" if delta <= 0 else "#dc3545"
        delta_symbol = "↓" if delta <= 0 else "↑"
        delta_html = f"""
        <div style="font-size: 0.875rem; color: {delta_color}; margin-top: 0.25rem;">
            {delta_symbol} {abs(delta)}{unit}
        </div>
        """
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border: 1px solid {border_color};
        border-left: 4px solid {border_color};
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    ">
        <div style="color: {text_color}; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.25rem;">
            {label}
        </div>
        <div style="color: {text_color}; font-size: 1.5rem; font-weight: 600; line-height: 1.2;">
            {formatted_value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


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
    
    # Just render the table directly without wrapper divs
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
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 0.75rem;
        border-radius: 0.375rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Force all text in metrics to be dark */
    .stMetric * {
        color: #212529 !important;
    }
    
    .stMetric > div > div {
        color: #212529 !important;
        font-weight: 500;
    }
    
    .stMetric > div > div[data-testid="metric-container"] > div {
        color: #212529 !important;
    }
    
    .stMetric [data-testid="metric-container"] * {
        color: #212529 !important;
    }
    
    .metric-good {
        border-left: 4px solid #28a745;
        background-color: #d4edda;
    }
    
    .metric-good * {
        color: #155724 !important;
    }
    
    .metric-bad {
        border-left: 4px solid #dc3545;
        background-color: #f8d7da;
    }
    
    .metric-bad * {
        color: #721c24 !important;
    }
    
    .metric-neutral {
        border-left: 4px solid #6c757d;
        background-color: #f8f9fa;
    }
    
    .metric-neutral * {
        color: #212529 !important;
    }
    
    /* Remove unused stage-table and breach-alert styles since we removed the wrappers */
    
    /* Override Streamlit's default metric styling */
    div[data-testid="metric-container"] div {
        color: #212529 !important;
    }
    
    div[data-testid="metric-container"] > div:first-child {
        color: #495057 !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="metric-container"] > div:last-child {
        color: #212529 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)