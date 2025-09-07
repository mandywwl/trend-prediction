from datetime import datetime, timezone


def apply_time_decay(ts, reference_time, decay_factor=0.5):
    """Apply exponential decay based on the time difference from a reference point.
    Args:
        ts (datetime): The timestamp of the event.
        reference_time (datetime or str): The reference time for decay calculation.
        decay_factor (float): The decay factor to apply. Default is 0.5.
    """
    if isinstance(reference_time, str):
        reference_time = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))
    
    # Ensure both timestamps are timezone-aware
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    delta_hours = (reference_time - ts).total_seconds() / 3600.0
    return round(decay_factor**delta_hours, 4)

