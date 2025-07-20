## Setup Instructions

1. Create and activate your virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Install Python requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install Playwright browsers:**
    ```bash
    playwright install
    ```
    This is required for Playwright-based scrapers (e.g., YouTube, TikTok).  
    You only need to do this once after installing requirements.

4. Run the main script as usual:
    ```bash
    python src/main.py
    ```
