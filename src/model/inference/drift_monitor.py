"""Performance drift monitoring utilities.

This module provides :class:`PerformanceDriftMonitor` which tracks rolling
F1 and Precision@K metrics for model predictions.  An alert callback is
invoked whenever the rolling averages fall below configured thresholds.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable, Sequence


class PerformanceDriftMonitor:
    """Monitor rolling F1 and Precision@K values.

    The monitor stores the most recent ``window`` metric values and computes
    rolling averages.  When either average drops below the configured
    thresholds, the ``alert_fn`` is called with a descriptive message.
    """

    def __init__(
        self,
        *,
        window: int = 100,
        f1_threshold: float = 0.7,
        precision_k_threshold: float = 0.7,
        k: int = 5,
        alert_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.window = int(window)
        self.f1_threshold = float(f1_threshold)
        self.precision_k_threshold = float(precision_k_threshold)
        self.k = int(k)
        self.alert_fn = alert_fn or (lambda msg: print(msg))

        self._f1_scores: Deque[float] = deque(maxlen=self.window)
        self._p_at_k_scores: Deque[float] = deque(maxlen=self.window)

    # ------------------------------------------------------------------
    def update(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        *,
        topk_pred: Sequence[int],
        relevant_items: Iterable[int],
    ) -> None:
        """Update metric buffers and check for drift.

        Args:
            y_true: Ground-truth binary labels.
            y_pred: Predicted binary labels.
            topk_pred: Ordered item ids ranked by predicted score.
            relevant_items: Iterable of item ids considered relevant for
                computing Precision@K.
        """
        f1 = self._f1_score(y_true, y_pred)
        p_at_k = self._precision_at_k(topk_pred, relevant_items)
        self._f1_scores.append(f1)
        self._p_at_k_scores.append(p_at_k)
        self._check_alerts()

    # ------------------------------------------------------------------
    def _check_alerts(self) -> None:
        if self._f1_scores:
            avg_f1 = sum(self._f1_scores) / len(self._f1_scores)
            if avg_f1 < self.f1_threshold:
                self.alert_fn(
                    f"Rolling F1 dropped to {avg_f1:.3f} (< {self.f1_threshold:.3f})"
                )
        if self._p_at_k_scores:
            avg_p = sum(self._p_at_k_scores) / len(self._p_at_k_scores)
            if avg_p < self.precision_k_threshold:
                self.alert_fn(
                    f"Rolling Precision@{self.k} dropped to {avg_p:.3f} (< {self.precision_k_threshold:.3f})"
                )

    # ------------------------------------------------------------------
    def _f1_score(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        tp = fp = fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    def _precision_at_k(
        self, predictions: Sequence[int], relevant_items: Iterable[int]
    ) -> float:
        k = min(self.k, len(predictions))
        if k == 0:
            return 0.0
        rel_set = set(relevant_items)
        hits = 0
        for item in list(predictions)[:k]:
            if item in rel_set:
                hits += 1
        return hits / float(k)
