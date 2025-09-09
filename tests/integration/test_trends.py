import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_pipeline.collectors.google_trends_collector import start_google_trends_collector

def test_event(event):
    print(f"GOT TRENDS EVENT: {event}")

print("Testing Google Trends collector...")
try:
    # Test with very short interval
    start_google_trends_collector(
        on_event=test_event,
        region="US",
        category="all", 
        count=5,
        interval=10  # 10 seconds for testing
    )
except KeyboardInterrupt:
    print("Test stopped")
except Exception as e:
    print(f"Error: {e}")