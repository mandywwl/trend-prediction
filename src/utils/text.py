import re


def slugify(text: str) -> str:
    """Convert text to safe identifier."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")