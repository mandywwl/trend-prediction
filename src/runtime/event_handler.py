"""Event processing utilities for the streaming runtime."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Sequence

import numpy as np

from embeddings.rt_distilbert import RealtimeTextEmbedder
from robustness.spam_filter import SpamScorer
from robustness.adaptive_thresholds import SensitivityController
from config.config import EMBED_PREPROC_BUDGET_MS
from config.schemas import Event, Features


class EmbeddingPreprocessor:
    """Attach text embeddings to incoming events.

    The preprocessor looks for known text fields, encodes the first one found
    using :class:`RealtimeTextEmbedder` and stores the resulting vector under
    ``event["features"]["text_emb"]``.
    """

    def __init__(
        self,
        embedder: RealtimeTextEmbedder,
        *,
        text_fields: Sequence[str] | None = None,
    ) -> None:
        """Create a new :class:`EmbeddingPreprocessor`.

        Args:
            embedder: Instance of :class:`RealtimeTextEmbedder` used for
                generating embeddings.
            text_fields: Optional list of event keys to inspect for text. The
                first present key is used. Defaults to common fields such as
                ``"text"``, ``"tweet_text"`` and ``"caption"``.
        """
        self.embedder = embedder
        self.text_fields = (
            list(text_fields)
            if text_fields is not None
            else [
                "text",
                "tweet_text",
                "caption",
                "description",
            ]
        )

        self._zero_emb: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _extract_text(self, event: Event) -> str | None:
        for field in self.text_fields:
            value = event.get(field)
            if isinstance(value, str) and value.strip():
                return value
        return None

    # ------------------------------------------------------------------
    def __call__(self, event: Event, *, light: bool = False) -> Event:
        """Process ``event`` in-place and return it.

        If no text field is found a zero embedding is attached to satisfy the
        downstream schema.
        """
        t0 = time.perf_counter()
        text = self._extract_text(event)
        if text is None or light:
            if self._zero_emb is None:
                dim = self.embedder.model.config.hidden_size
                self._zero_emb = np.zeros(dim, dtype=np.float32)
            emb = self._zero_emb
        else:
            emb = self.embedder.encode([text])[0]

        features: Features = event.setdefault("features", {})
        features["text_emb"] = emb

        duration_ms = (time.perf_counter() - t0) * 1000.0
        if duration_ms > EMBED_PREPROC_BUDGET_MS:
            print(
                f"[EmbeddingPreprocessor] budget exceeded: {duration_ms:.1f}ms > {EMBED_PREPROC_BUDGET_MS}ms"
            )
        return event


class EventHandler:
    """Handle events by preprocessing then invoking TGN inference."""

    def __init__(
        self,
        embedder: RealtimeTextEmbedder,
        infer: Callable[[Dict[str, Any]], Any],
        *,
        spam_scorer: SpamScorer | None = None,
        sensitivity: "SensitivityController | None" = None,
    ) -> None:
        """Initialise the handler.

        Args:
            embedder: Text embedder used by the ``EmbeddingPreprocessor``.
            infer: Callable representing the TGN inference service. It receives
                the processed event.
            spam_scorer: Optional :class:`SpamScorer` used to down-weight
                edges for suspected spammy accounts.
        """
        self.preprocessor = EmbeddingPreprocessor(embedder)
        self._infer = infer
        self.spam_scorer = spam_scorer
        self.sensitivity = sensitivity

    # ------------------------------------------------------------------
    def handle(self, event: Event) -> Any:
        """Preprocess ``event`` and forward it to inference.

        If a sensitivity controller is provided and currently applying
        back-pressure, heavy operations such as embeddings are skipped by
        attaching a zero-vector. The handler also records observed latency
        into the controller for adaptive behaviour.
        """

        policy = None
        if self.sensitivity is not None:
            policy = self.sensitivity.policy()
        light = bool(policy and not policy.heavy_ops_enabled)

        t0 = time.perf_counter()
        processed = self.preprocessor(event, light=light)

        if self.spam_scorer is not None:
            weight = self.spam_scorer.edge_weight(processed)
            processed.setdefault("features", {})["edge_weight"] = weight

        result = self._infer(processed)

        # Record latency and (optional) ground-truth spam label for adaptation
        if self.sensitivity is not None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            is_spam = bool(processed.get("is_spam", False))
            try:
                self.sensitivity.record_event(is_spam=is_spam, latency_ms=latency_ms)
            except Exception:
                pass  # never let adaptation interfere with the hot path

            # Expose suggested sampler size for downstream components
            pol = self.sensitivity.policy()
            processed.setdefault("features", {})["sampler_size"] = pol.sampler_size

        return result
