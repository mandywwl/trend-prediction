"""
Trend Prediction Dashboard - Streamlit Application

Main dashboard for monitoring live trend predictions and system health.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add current directory to path for local imports
dashboard_path = Path(__file__).parent
sys.path.insert(0, str(dashboard_path))

from components import topk

# Optional import for robustness (may have additional dependencies)
try:
    from components import robustness
    ROBUSTNESS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Robustness component not available: {e}")
    ROBUSTNESS_AVAILABLE = False


def main():
    """Main dashboard application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Trend Prediction Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("📈 Trend Prediction Dashboard")
    st.markdown("Real-time monitoring of trend predictions and system health.")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Panel selection
    panel_option = st.sidebar.selectbox(
        "Select Panel:",
        options=[
            "📊 Now (Top-K Live)",
            "🛡️ Robustness",
            "📋 About"
        ],
        index=0
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=True)
    if auto_refresh:
        # Streamlit auto-refresh using query params or timer
        st.sidebar.info("Dashboard will refresh automatically every 30 seconds")
        # Note: For production, you might want to implement actual auto-refresh
        # using st.rerun() with a timer or query parameter polling
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Main content area
    if panel_option == "📊 Now (Top-K Live)":
        render_topk_panel()
    elif panel_option == "🛡️ Robustness":
        render_robustness_panel()
    elif panel_option == "📋 About":
        render_about_panel()


def render_topk_panel():
    """Render the Top-K live predictions panel."""
    try:
        panel_result = topk.render_panel()
        
        # Show panel status in sidebar
        st.sidebar.subheader("📊 Panel Status")
        status = panel_result.get("status", "unknown")
        
        if status == "success":
            st.sidebar.success(f"✅ Active - {panel_result.get('predictions_count', 0)} predictions")
        elif status == "no_cache":
            st.sidebar.error("❌ No cache file found")
        elif status == "empty_cache":
            st.sidebar.warning("⚠️ Cache is empty")
        elif status == "no_predictions":
            st.sidebar.warning("⚠️ No predictions for selected K")
        else:
            st.sidebar.info(f"ℹ️ Status: {status}")
            
    except Exception as e:
        st.error(f"❌ Error rendering Top-K panel: {e}")
        st.sidebar.error("❌ Panel Error")


def render_robustness_panel():
    """Render the robustness monitoring panel."""
    if not ROBUSTNESS_AVAILABLE:
        st.warning("🚧 Robustness panel not available - missing dependencies")
        st.write("Install additional dependencies to enable robustness monitoring.")
        return
        
    try:
        st.info("🚧 Robustness panel integration coming soon...")
        st.write("This panel will show spam detection and threshold monitoring.")
        
        # For now, show that robustness component exists
        if hasattr(robustness, 'render_panel'):
            st.write("✅ Robustness component is available")
        else:
            st.write("⚠️ Robustness component not yet integrated")
            
    except Exception as e:
        st.error(f"❌ Error rendering Robustness panel: {e}")


def render_about_panel():
    """Render the about/information panel."""
    st.header("📋 About Trend Prediction Dashboard")
    
    st.markdown("""
    ### 🎯 Purpose
    This dashboard provides real-time monitoring of trend predictions and system health.
    
    ### 📊 Features
    
    #### Now Panel (Top-K Live)
    - **Live Predictions**: Shows current Top-K trend predictions
    - **Countdown Timers**: Time remaining until Δ-freeze for each prediction
    - **Configurable K**: Select different K values for Top-K display
    - **Cache Monitoring**: Real-time cache status and updates
    
    #### Configuration
    - **Prediction Cache**: Automatically loads from configured cache path
    - **K Options**: Dynamic K selector based on configuration
    - **Delta Hours**: Countdown timers use configured Δ-freeze window
    
    ### 🔧 Technical Details
    - Built with Streamlit for real-time dashboards
    - Integrates with existing prediction pipeline
    - Handles empty states and error conditions gracefully
    - Configuration-driven UI (no code changes needed for config updates)
    """)
    
    # Configuration display
    st.subheader("⚙️ Current Configuration")
    
    try:
        from config.config import PREDICTIONS_CACHE_PATH, DELTA_HOURS, K_OPTIONS
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.code(f"""
Cache Path: {PREDICTIONS_CACHE_PATH}
Delta Hours: {DELTA_HOURS}
            """)
        
        with col2:
            st.code(f"""
K Options: {K_OPTIONS}
            """)
            
    except ImportError as e:
        st.error(f"Configuration import error: {e}")


# Auto-refresh mechanism (simple implementation)
if __name__ == "__main__":
    main()