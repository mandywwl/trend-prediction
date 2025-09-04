from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure the directory exists and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# TODO: Expand with additional I/O helpers if required.
