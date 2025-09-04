from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


# TODO: Provide further time utilities if needed.
