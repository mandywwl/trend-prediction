import copy
import logging
import random
from datetime import datetime, timedelta
from typing import Dict

from config import config
from config.schemas import Batch

logger = logging.getLogger(__name__)


def _jitter_ts(ts_iso: str, rng: random.Random) -> str:
    jitter_min = config.NOISE_JITTER_MIN_MIN
    jitter_max = config.NOISE_JITTER_MAX_MIN
    minutes = rng.uniform(jitter_min, jitter_max)
    sign = rng.choice([-1, 1])
    delta = timedelta(minutes=sign * minutes)
    dt = datetime.fromisoformat(ts_iso)
    return (dt + delta).isoformat()


def inject_noise(
    batch: Batch, p: float | None = None, seed: int | None = None
) -> Batch:
    """Inject synthetic noise into a batch of events.

    Parameters
    ----------
    batch:
        Sequence of events to corrupt.
    p:
        Fraction of events to corrupt. Defaults to ``config.NOISE_P_DEFAULT``.
    seed:
        Optional RNG seed for reproducibility.
    """
    if p is None:
        p = config.NOISE_P_DEFAULT
    if not batch or p <= 0:
        return batch

    rng = random.Random(seed)
    n = len(batch)
    k = max(1, int(p * n))
    indices = rng.sample(range(n), k)

    new_batch = list(batch)
    counts: Dict[str, int] = {"missing": 0, "duplicate": 0, "jitter": 0, "bot_burst": 0}

    for idx in indices:
        event = copy.deepcopy(new_batch[idx])
        mode = rng.choice(list(counts.keys()))
        if mode == "missing":
            event.get("features", {}).pop("text_emb", None)
            new_batch[idx] = event
            counts[mode] += 1
        elif mode == "duplicate":
            new_batch.append(event)
            counts[mode] += 1
        elif mode == "jitter":
            event["ts_iso"] = _jitter_ts(event["ts_iso"], rng)
            new_batch[idx] = event
            counts[mode] += 1
        elif mode == "bot_burst":
            burst = rng.randint(2, 4)
            for _ in range(burst):
                dup = copy.deepcopy(event)
                dup["ts_iso"] = _jitter_ts(event["ts_iso"], rng)
                new_batch.append(dup)
                counts[mode] += 1

    logger.info("noise injection counts: %s", counts)
    return new_batch
