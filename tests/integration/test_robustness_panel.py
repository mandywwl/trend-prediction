"""Quick test script to verify robustness panel functionality."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add dashboard to path  
dashboard_path = Path(__file__).parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from dashboard.components import robustness
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_robustness_panel():
    """Test the robustness panel component."""
    print("Testing robustness panel...")
    
    try:
        # Call the render_panel function
        result = robustness.render_panel(datasets_dir="datasets")
        
        print(f"Result keys: {list(result.keys())}")
        print(f"Spam rate: {result.get('spam_rate')}")
        print(f"Down-weighted %: {result.get('downweighted_pct')}")
        
        alert = result.get('alert', {})
        print(f"Alert level: {alert.get('level')}")
        print(f"Alert message: {alert.get('message')}")
        
        figure = result.get('figure')
        if figure:
            print("✅ Figure generated successfully")
            # Save the figure to verify it works
            figure.savefig("test_robustness_plot.png")
            plt.close(figure)
            print("✅ Figure saved as test_robustness_plot.png")
        else:
            print("❌ No figure generated")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing robustness panel: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_robustness_panel()
    if success:
        print("✅ Robustness panel test completed successfully")
    else:
        print("❌ Robustness panel test failed")
