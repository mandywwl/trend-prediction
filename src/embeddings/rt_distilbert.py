"""Realtime DistilBERT text embedder with micro-batching and caching."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from queue import Queue, Empty
from typing import List

import numpy as np
import torch
from cachetools import TTLCache
from transformers import AutoModel, AutoTokenizer


class RealtimeTextEmbedder:
    """Encode text to embeddings using DistilBERT with real-time features.

    The embedder keeps a worker thread that batches incoming requests to
    amortise model execution cost. Repeated texts are memoised in an LRU cache
    with TTL to minimise latency.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        *,
        device: str | None = None,
        batch_size: int = 8,
        max_latency_ms: int = 50,
        cache_maxsize: int = 1024,
        cache_ttl: float = 300.0,
    ) -> None:
        """Initialise the embedder and load DistilBERT once.

        Args:
            model_name: HuggingFace model identifier.
            device: Target device ("cpu", "cuda" or "mps"). Fallbacks to CPU
                if unavailable.
            batch_size: Maximum batch size processed by the worker.
            max_latency_ms: Maximum time the worker waits to form a batch.
            cache_maxsize: Maximum entries to keep in the LRU cache.
            cache_ttl: Time-to-live for cache entries in seconds.
        """

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        elif device == "mps" and not getattr(torch.backends, "mps", None):
            device = "cpu"

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_latency_ms = max_latency_ms
        self.cache: TTLCache[str, np.ndarray] = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.queue: Queue[tuple[str, Future]] = Queue()

        # Model and tokenizer are loaded once during init.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Background worker thread.
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of input strings.

        Returns:
            Array of shape (len(texts), hidden_dim) containing mean pooled
            embeddings.
        """

        results: list[np.ndarray | None] = [None] * len(texts)
        futures: list[Future] = []
        pending_idx: list[int] = []

        for idx, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[idx] = cached
            else:
                fut: Future[np.ndarray] = Future()
                self.queue.put((text, fut))
                futures.append(fut)
                pending_idx.append(idx)

        for fut, idx in zip(futures, pending_idx):
            emb = fut.result()
            results[idx] = emb
            self.cache[texts[idx]] = emb

        return np.vstack(results)

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        """Background thread consuming requests and performing encoding."""
        while True:
            text, fut = self.queue.get()
            batch: list[tuple[str, Future]] = [(text, fut)]
            start = time.perf_counter()
            while len(batch) < self.batch_size:
                remaining = self.max_latency_ms / 1000 - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except Empty:
                    break

            texts = [t for t, _ in batch]
            backlog = self.queue.qsize()

            device = self.device
            max_length: int | None = None
            if backlog > self.batch_size * 4:
                max_length = 32
                if device.type != "cpu":
                    device = torch.device("cpu")

            embeddings = self._encode_batch(texts, device=device, max_length=max_length)
            for emb, (_, f) in zip(embeddings, batch):
                f.set_result(emb)

    # ------------------------------------------------------------------
    def _encode_batch(
        self,
        texts: List[str],
        *,
        device: torch.device,
        max_length: int | None,
    ) -> np.ndarray:
        """Encode a batch of texts on the specified device."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            hidden = outputs.last_hidden_state
            embeddings = hidden.mean(dim=1).cpu().numpy()
        return embeddings
