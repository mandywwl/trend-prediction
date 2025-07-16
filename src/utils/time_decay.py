from datetime import datetime

def apply_time_decay(ts, reference_time, decay_factor=0.5):
    if isinstance(reference_time, str):
        reference_time = datetime.fromisoformat(reference_time)
    delta_hours = (ts - reference_time).total_seconds() / 3600.0
    return round(decay_factor ** delta_hours, 4)