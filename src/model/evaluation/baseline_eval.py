"""Utilities for evaluating baseline models."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List

import numpy as np

from model.baselines import SnapshotLSTM, GPPHeuristic
from model.evaluation.metrics import PrecisionAtKOnline


def evaluate_baselines(
    sequences: Dict[int, Iterable[float]],
    emergent_topics: List[int],
    ts_iso: str,
) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline models on simple sequence data.

    Parameters
    ----------
    sequences: mapping of ``topic_id -> sequence`` where each sequence is an
        iterable of numeric values representing hourly counts.
    emergent_topics: list of topic IDs that should be marked as emergent for
        metric calculation.
    ts_iso: timestamp of the prediction in ISO format.

    Returns
    -------
    Dict mapping model name to the snapshot from
    :class:`~model.evaluation.metrics.PrecisionAtKOnline`.
    """

    models = {
        "snapshot_lstm": SnapshotLSTM(input_size=1, hidden_size=4),
        "gpp": GPPHeuristic(window=1),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        evaluator = PrecisionAtKOnline()
        preds = []
        for topic_id, seq in sequences.items():
            if isinstance(model, SnapshotLSTM):
                arr = np.asarray(seq, dtype=float).reshape(1, -1, 1)
                score = float(model.forward(arr)[0])
            else:
                score = float(model.predict(list(seq)))
            preds.append((topic_id, score))
        evaluator.record_predictions(ts_iso=ts_iso, items=preds)
        for tid in emergent_topics:
            evaluator.record_event(topic_id=tid, user_id=f"u{tid}", ts_iso=ts_iso)
        # Finalize by pushing time forward so entries mature.
        # ``PrecisionAtKOnline`` uses DELTA_HOURS=2 by default, so advance
        # beyond that horizon to ensure support > 0.
        now = datetime.fromisoformat(ts_iso) + timedelta(hours=3)
        evaluator.record_event(
            topic_id=999, user_id="noop", ts_iso=now.isoformat(timespec="seconds")
        )
        results[name] = evaluator.rolling_hourly_scores()
    return results
