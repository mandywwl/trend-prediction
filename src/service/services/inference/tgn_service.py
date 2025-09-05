"""Minimal production-ready TGN inference service.

Wraps a trained Temporal Graph Network (TGN) for online inference in an
event-driven pipeline. Each event updates the temporal memory and triggers a
forward pass that yields per-topic trend emergence scores in [0, 1].

Key features:
- LRU-based soft memory limiting via MAX_NODES.
- Missing text embeddings handled via DEFAULT_TEXT_EMB_POLICY.
- Append-only logging of update and forward latencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json
import time

import numpy as np
import torch

from config.config import (
    MAX_NODES,
    DEFAULT_TEXT_EMB_POLICY,
)
from config.schemas import Event
from model.core.tgn import TGNModel
from utils.io import ensure_dir


@dataclass
class _NodeMap:
    """Bidirectional mapping between external node ids and memory indices.

    Maintains an LRU order (via dict insertion order) for eviction. Python 3.7+
    preserves insertion order; we use pop/insert to move to end on access.
    """

    id_to_idx: Dict[str, int]
    idx_to_id: Dict[int, str]

    def touch(self, node_id: str, idx: int) -> None:
        # Move to end to mark as recently used
        if node_id in self.id_to_idx:
            self.id_to_idx.pop(node_id, None)
        self.id_to_idx[node_id] = idx
        self.idx_to_id[idx] = node_id

    def pop_lru(self) -> tuple[str, int] | None:
        if not self.id_to_idx:
            return None
        # pop first inserted (least recently used)
        node_id, idx = next(iter(self.id_to_idx.items()))
        self.id_to_idx.pop(node_id, None)
        self.idx_to_id.pop(idx, None)
        return node_id, idx


class TGNInferenceService:
    """Online inference wrapper for a trained TGN.

    Usage:
        service = TGNInferenceService(checkpoint_path="datasets/tgn_model.pt")
        scores = service.update_and_score(event)
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        memory_dim: int = 100,
        time_dim: int = 10,
        edge_feat_dim: int | None = None,
        log_dir: str | Path = "datasets",
    ) -> None:
        """Initialize the inference service and load a checkpoint if provided.

        Args:
            checkpoint_path: Optional path to model checkpoint (.pt/.pth) with
                a compatible state_dict.
            device: Target device. Defaults to CPU for predictable latency.
            memory_dim: Memory dimension used by TGN.
            time_dim: Temporal encoding dimension.
            edge_feat_dim: Expected edge feature dimension (e.g., text_emb
                size). If None, defaults to 768 (DistilBERT hidden size).
            log_dir: Directory where append-only logs are written.
        """
        self.device = torch.device(device)

        # Default embedding dimension to DistilBERT hidden size when unknown
        self.edge_dim = int(edge_feat_dim or 768)

        # Pre-create zero and running mean embeddings for fallback policies
        self._zero_emb = torch.zeros(self.edge_dim, dtype=torch.float32)
        self._mean_emb = torch.zeros(self.edge_dim, dtype=torch.float32)
        self._mean_count = 0

        # Learned unknown embedding (used when DEFAULT_TEXT_EMB_POLICY says so)
        self._learned_unknown = torch.nn.Parameter(
            torch.zeros(self.edge_dim, dtype=torch.float32)
        )

        # Model is sized for MAX_NODES capacity; internal mapping enforces LRU
        self.model = TGNModel(
            num_nodes=MAX_NODES,
            node_feat_dim=0,
            edge_feat_dim=self.edge_dim,
            time_dim=time_dim,
            memory_dim=memory_dim,
        ).to(self.device)
        # Register learned unknown on same device in case used in forward
        self._learned_unknown = self._learned_unknown.to(self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # Mapping from external ids to memory indices with LRU eviction
        self._nodes = _NodeMap(id_to_idx={}, idx_to_id={})
        self._free_indices = list(range(MAX_NODES))

        # Append-only latency log path
        ensure_dir(Path(log_dir))
        self._latency_log = Path(log_dir) / "tgn_inference.log"

        # Small scratch buffers to avoid reallocations
        self._scratch = {
            "src": None,
            "dst": None,
            "t": None,
        }

        # Torch settings for inference
        self.model.eval()
        torch.set_grad_enabled(False)

    # ------------------------------------------------------------------
    def _load_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        state = torch.load(p, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Support both strict and partial load to be tolerant to wrapper keys
        try:
            self.model.load_state_dict(state, strict=True)
        except Exception:
            # Try to strip any leading module/ prefixes
            cleaned = {k.split("module.")[-1]: v for k, v in state.items()}
            self.model.load_state_dict(cleaned, strict=False)

    # ------------------------------------------------------------------
    def _parse_ts(self, ts_iso: str) -> float:
        try:
            # Python can parse Z if replaced
            from datetime import datetime

            s = ts_iso.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return float(dt.timestamp())
        except Exception:
            # Fallback to monotonic-ish timestamp
            return time.time()

    # ------------------------------------------------------------------
    def _get_or_assign_idx(self, node_id: str) -> int:
        idx = self._nodes.id_to_idx.get(node_id)
        if idx is not None:
            self._nodes.touch(node_id, idx)
            return idx

        # Need a new slot: from free list or evict LRU
        if self._free_indices:
            idx = self._free_indices.pop()
        else:
            evicted = self._nodes.pop_lru()
            if evicted is None:
                # Should not happen; create index 0 fallback
                idx = 0
            else:
                _, idx = evicted
                # Reset memory state for that index
                self._reset_node_state(idx)
        self._nodes.touch(node_id, idx)
        return idx

    # ------------------------------------------------------------------
    def _reset_node_state(self, idx: int) -> None:
        # Zero memory and last_update for a single node index
        mem = self.model.memory.memory
        last = self.model.memory.last_update
        if idx < mem.size(0):
            mem[idx].zero_()
        if idx < last.size(0):
            last[idx] = 0

    # ------------------------------------------------------------------
    def _edge_attr_from_features(self, features: dict | None) -> torch.Tensor:
        if not features:
            policy = DEFAULT_TEXT_EMB_POLICY
            if policy == "mean" and self._mean_count > 0:
                return self._mean_emb.to(self.device)
            if policy == "learned_unknown":
                return self._learned_unknown
            return self._zero_emb.to(self.device)

        emb = features.get("text_emb") if isinstance(features, dict) else None

        if emb is None:
            policy = DEFAULT_TEXT_EMB_POLICY
            if policy == "mean" and self._mean_count > 0:
                return self._mean_emb.to(self.device)
            if policy == "learned_unknown":
                return self._learned_unknown
            return self._zero_emb.to(self.device)

        # Convert to torch vector of expected dim
        if isinstance(emb, np.ndarray):
            v = torch.from_numpy(emb.astype(np.float32, copy=False))
        else:
            v = torch.as_tensor(emb, dtype=torch.float32)

        if v.numel() != self.edge_dim:
            # Pad/truncate to expected dim for robustness
            if v.numel() < self.edge_dim:
                pad = torch.zeros(self.edge_dim - v.numel(), dtype=torch.float32)
                v = torch.cat([v.flatten(), pad], dim=0)
            else:
                v = v.flatten()[: self.edge_dim]

        # Update running mean (Welford simplified for vectors)
        with torch.no_grad():
            self._mean_count += 1
            delta = v - self._mean_emb
            self._mean_emb += delta / float(self._mean_count)

        return v.to(self.device)

    # ------------------------------------------------------------------
    def _log_latencies(self, update_ms: float, forward_ms: float) -> None:
        try:
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "update_ms": round(float(update_ms), 3),
                "forward_ms": round(float(forward_ms), 3),
            }
            with self._latency_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            # Never block or fail on logging
            pass

    # ------------------------------------------------------------------
    def update_and_score(self, event: Event) -> Dict[str, float]:
        """Update the TGN with the event and return per-topic scores.

        The service processes only the first target in ``target_ids`` for the
        forward pass to preserve latency; additional targets can be handled by
        subsequent events if required upstream.

        Args:
            event: Event dict with keys {event_id, ts_iso, actor_id,
                target_ids, edge_type, features}.

        Returns:
            Mapping of topic identifier to score in [0, 1]. Currently a
            single-topic output is provided under key "topic:0".
        """
        actor_id = str(event.get("actor_id", "0"))
        targets = event.get("target_ids") or []
        target_id = str(targets[0] if targets else actor_id)
        t_iso = str(event.get("ts_iso", ""))

        # Resolve memory indices (LRU eviction if needed)
        src_idx = self._get_or_assign_idx(actor_id)
        dst_idx = self._get_or_assign_idx(target_id)

        # Build tensors
        t_val = self._parse_ts(t_iso)
        t_tensor = torch.tensor([t_val], dtype=torch.float32, device=self.device)
        src_tensor = torch.tensor([src_idx], dtype=torch.long, device=self.device)
        dst_tensor = torch.tensor([dst_idx], dtype=torch.long, device=self.device)
        e_attr = self._edge_attr_from_features(event.get("features"))
        e_attr = e_attr.view(1, -1).to(self.device)

        # Update memory state: detach graph then update
        up_start = time.perf_counter()
        try:
            self.model.memory.detach()
        except Exception:
            pass
        # TGNMemory expects integer timestamps for update_state
        t_event = t_tensor.long()
        self.model.memory.update_state(src_tensor, dst_tensor, t_event, e_attr)
        update_ms = (time.perf_counter() - up_start) * 1000.0

        # Forward pass
        fwd_start = time.perf_counter()
        logits = self.model(src_tensor, dst_tensor, t_tensor, e_attr)
        probs = torch.sigmoid(logits).view(-1)
        forward_ms = (time.perf_counter() - fwd_start) * 1000.0

        # Log latencies (append-only, non-blocking)
        self._log_latencies(update_ms, forward_ms)

        # Produce per-topic score dictionary (single topic for now)
        score = float(probs[0].clamp(0.0, 1.0).item())
        return {"topic:0": score}
