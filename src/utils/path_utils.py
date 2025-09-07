from pathlib import Path


def find_repo_root(start: Path | None = None, marker: str = "pyproject.toml") -> Path:
    """Walk upwards from ``start`` until a folder containing ``marker`` is found.

    Args:
        start: Optional starting path. Defaults to the location of this file.
        marker: Filename used to identify the repository root.

    Returns:
        The repository root as a :class:`Path`.
    """
    p = (start or Path(__file__).resolve()).parent
    for candidate in [p, *p.parents]:
        if (candidate / marker).exists():
            return candidate
    return p  # fallback if marker not found
