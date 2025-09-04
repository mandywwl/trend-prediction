from datetime import datetime


def apply_time_decay(ts, reference_time, decay_factor=0.5):
    """Apply exponential decay based on the time difference from a reference point.
    Args:
        ts (datetime): The timestamp of the event.
        reference_time (datetime or str): The reference time for decay calculation.
        decay_factor (float): The decay factor to apply. Default is 0.5.
    """
    if isinstance(reference_time, str):
        reference_time = datetime.fromisoformat(reference_time)

    delta_hours = (reference_time - ts).total_seconds() / 3600.0
    return round(decay_factor**delta_hours, 4)
