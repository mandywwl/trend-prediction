"""Minimal production-ready TGN inference service.

Wraps a trained Temporal Graph Network (TGN) for online inference in an
event-driven pipeline. Each event updates the temporal memory and triggers a
forward pass that yields a single **predicted growth score** used for ranking.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json
import time
import logging
import numpy as np
# import math
import torch

from model.core.tgn import TGNModel
from config.config import (
    MAX_NODES,
    TEXT_EMB_POLICY,
    TGN_MEMORY_DIM,
    TGN_TIME_DIM,
    INFERENCE_DEVICE,
    TGN_EDGE_DIM,
    GROWTH_FACTOR_BASE,
    DELTA_HOURS,

)
from config.schemas import Event
from utils.datetime import timestamp_to_seconds
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
        device: str | torch.device = INFERENCE_DEVICE,
        memory_dim: int = TGN_MEMORY_DIM,
        time_dim: int = TGN_TIME_DIM,
        edge_feat_dim: int = TGN_EDGE_DIM,
        log_dir: str | Path = "datasets",
    ) -> None:
        """Initialize the inference service and load a checkpoint if provided."""

        self.logger = logging.getLogger(__name__)
        self.device = torch.device(device)
        self.edge_dim = int(edge_feat_dim or TGN_EDGE_DIM or 768)
        self.policy = TEXT_EMB_POLICY

        # # Online calibraters for growth and diffusion heads
        # self._ema_growth_mu = 0.0
        # self._ema_growth_var = 1.0
        # self._ema_diff_mu = 0.0
        # self._ema_diff_var = 1.0
        # self._ema_beta = 0.98 # smoothing factor

        # Optional online calibration for the single growth signal
        self._ema_growth_mu = 0.0
        self._ema_growth_var = 1.0
        self._ema_beta = 0.98  # smoothing factor

        ckpt_hparams = {}

        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            if p.exists():
                try:
                    state_dict, saved_hparams = self._peek_state(p)
                    # If trainer saved hparams, prefer them
                    if isinstance(saved_hparams, dict):
                        ckpt_hparams = dict(saved_hparams)
                    else:
                        ckpt_hparams = self._infer_hparams(state_dict, default_time_dim=TGN_TIME_DIM)
                except Exception as e:
                    self.logger.warning(f"Could not peek checkpoint hparams ({p}): {e}")

        # Choose dims (ckpt -> config fallback)
        num_nodes  = int(ckpt_hparams.get("num_nodes", MAX_NODES))
        memory_dim = int(ckpt_hparams.get("memory_dim", TGN_MEMORY_DIM))
        time_dim   = int(ckpt_hparams.get("time_dim",   TGN_TIME_DIM))
        edge_dim   = int(ckpt_hparams.get("edge_feat_dim", TGN_EDGE_DIM))
        self.logger.info(
            "TGN dims → num_nodes=%d, memory_dim=%d, time_dim=%d, edge_dim=%d",
            num_nodes, memory_dim, time_dim, edge_dim,
        )

        self.edge_dim = edge_dim  # keep service’s expectation in sync

        # Pre-create zero and running mean embeddings for fallback policies
        self._zero_emb = torch.zeros(self.edge_dim, dtype=torch.float32)
        self._mean_emb = torch.zeros(self.edge_dim, dtype=torch.float32)
        self._mean_count = 0

        # Learned unknown embedding (used when  TEXT_EMB_POLICY says so)
        self._learned_unknown = torch.nn.Parameter(torch.zeros(self.edge_dim, dtype=torch.float32))

        # Instantiate the model with the derived sizes
        self.model = TGNModel(
            num_nodes=num_nodes,
            node_feat_dim=0,
            edge_feat_dim=self.edge_dim,
            time_dim=time_dim,
            memory_dim=memory_dim
        ).to(self.device)

        self._learned_unknown = self._learned_unknown.to(self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # Mapping from external ids to memory indices
        self._nodes = _NodeMap(id_to_idx={}, idx_to_id={})
        capacity = int(self.model.memory.memory.size(0))
        self._free_indices = list(range(capacity))

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
    
    def _peek_state(self, path: Path):
        obj = torch.load(path, map_location="cpu")
        # Support either {"state_dict": ...} or raw state_dict
        state_dict = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        return state_dict, (obj.get("hparams") if isinstance(obj, dict) else None)
    
    # ------------------------------------------------------------------

    def _infer_hparams(self, state_dict: dict, default_time_dim: int) -> dict:
        """Infer num_nodes, memory_dim, edge_feat_dim (and optionally time_dim) from a state_dict."""
        hp = {}
        mem = state_dict.get("memory.memory")
        if mem is not None:
            hp["num_nodes"] = int(mem.shape[0])
            hp["memory_dim"] = int(mem.shape[1])
        # Input size to GRU = memory_dim + edge_feat_dim + time_dim (in this implementation)
        wih = state_dict.get("memory.gru.weight_ih")
        if wih is not None and "memory_dim" in hp:
            input_size = int(wih.shape[1])
            # Use configured time_dim as a hint; compute edge_feat_dim from it
            time_dim = default_time_dim
            edge_feat_dim = max(1, input_size - hp["memory_dim"] - time_dim)
            hp["time_dim"] = time_dim
            hp["edge_feat_dim"] = edge_feat_dim
        return hp

    # ------------------------------------------------------------------
    def _load_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        obj = torch.load(p, map_location=self.device)
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

        # Try strict first
        try:
            self.model.load_state_dict(state, strict=True)
            return
        except Exception as e:
            self.logger.warning(f"Strict load failed, retrying with shape-filtered keys: {e}")

        
        # Filter to keys whose shapes match the current model
        current = self.model.state_dict()
        filtered = {}
        for k, v in state.items():
            kk = k.split("module.")[-1]
            if kk in current and current[kk].shape == v.shape:
                filtered[kk] = v
        missing = [k for k in current.keys() if k not in filtered]
        if missing:
            self.logger.info(f"Skipping {len(missing)} keys with mismatched shapes (ok)")

        self.model.load_state_dict(filtered, strict=False)

    # ------------------------------------------------------------------
    def reload_checkpoint(self, path: str | Path) -> None:
        """Hot-reload weights at runtime."""
        self._load_checkpoint(path)
        self.model.eval()


    # ------------------------------------------------------------------
    def _parse_ts(self, ts_iso: str) -> float:
        try:
            return timestamp_to_seconds(ts_iso)
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
        # existing: resolve v as the embedding vector 'text_emb' (float tensor)
        policy = TEXT_EMB_POLICY
        if not features:
            if policy == "mean" and self._mean_count > 0: return self._mean_emb.to(self.device)
            if policy == "learned_unknown": return self._learned_unknown
            return self._zero_emb.to(self.device)

        emb = features.get("text_emb") if isinstance(features, dict) else None
        if emb is None:
            if policy == "mean" and self._mean_count > 0: return self._mean_emb.to(self.device)
            if policy == "learned_unknown": return self._learned_unknown
            return self._zero_emb.to(self.device)
        
        # numpy/torch to torch float1D
        v = torch.from_numpy(emb.astype(np.float32, copy=False)) if isinstance(emb, np.ndarray) else torch.as_tensor(emb, dtype=torch.float32)

        # Append edge_weight if present
        ew = None
        if isinstance(features, dict) and "edge_weight" in features:
            try:
                ew = float(features.get("edge_weight", 1.0))
            except Exception:
                ew = None
        if ew is not None and self.edge_dim >= 2:
            # reserve the last slot for the scalar, trim/pad emb to (edge_dim - 1)
            target = max(1, self.edge_dim - 1)
            if v.numel() < target:
                pad = torch.zeros(target - v.numel(), dtype=torch.float32)
                v = torch.cat([v.flatten(), pad], dim=0)
            else:
                v = v.flatten()[:target]
            v = torch.cat([v, torch.tensor([ew], dtype=torch.float32)], dim=0)
        else:
            # fallback
            if v.numel() != self.edge_dim:
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

    def _ema_update(self, x: float, mu: float, var: float) -> tuple[float, float]:
        """Update (mu, var) with exponential smoothing."""
        beta = float(self._ema_beta)
        dx = x - mu
        mu = mu + (1.0 - beta) * dx
        var = beta * var + (1.0 - beta) * (dx * dx)
        return mu, var

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


    # --------------------------------------------
    def score_items_at_time(
        self,
        items: list[str] | tuple[str, ...],
        t_end_s: int | float,
        *,
        features: dict | None = None,
        mutate: bool = False,
    ) -> Dict[str, float]:
        """
        Score a list of item_ids at timestamp t_end_s using current memory.
        If mutate=False (default), the method DOES NOT call memory.update_state(),
        so it is safe to use during evaluation without changing model state.
        """
        if not items:
            return {}

        # Ensure stable "probe" actor exists; maps to a single node index
        probe_idx = self._get_or_assign_idx("__probe__")

        # Time & edge features (shared across batch)
        t_tensor = torch.tensor([float(t_end_s)], dtype=torch.float32, device=self.device)
        e_attr = self._edge_attr_from_features(features).view(1, -1)

        # Resolve / allocate destination indices (will reuse existing memory slots)
        dst_indices = [self._get_or_assign_idx(str(it)) for it in items]
        src = torch.tensor([probe_idx] * len(items), dtype=torch.long, device=self.device)
        dst = torch.tensor(dst_indices, dtype=torch.long, device=self.device)
        t = t_tensor.repeat(len(items))
        e = e_attr.repeat(len(items), 1)

        self.model.eval()
        with torch.no_grad():
            if mutate:
                # Mutating path (rarely needed in eval): update memory first
                self.model.memory.update_state(src, dst, t.long(), e)
            out = self.model(src, dst, t, e).view(-1) # Forward pass reads current memory state

        return {it: float(s) for it, s in zip(items, out.tolist())}
    # ---------------------------------------------------------------------------

    def update_and_score(self, event: Event) -> Dict[str, float]:
        """Update the TGN with the event and return the predicted growth score.

        The service processes only the first target in ``target_ids`` for the
        forward pass to preserve latency; additional targets can be handled by
        subsequent events if required upstream.

        Args:
            event: Event dict with keys {event_id, ts_iso, actor_id,
                target_ids, edge_type, features}.

        Returns:
            Mapping with at least:
              - ``growth_score``: scalar (float), higher = stronger predicted growth.
              - ``score``: alias of growth_score, for dashboards expecting "score".

        """
        actor_id = str(event.get("actor_id", "0"))
        targets = event.get("target_ids") or []
        target_id = str(targets[0] if targets else actor_id)
        t_iso = str(event.get("ts_iso", ""))

        # ------- Resolve memory indices (LRU eviction if needed) -------
        src_idx = self._get_or_assign_idx(actor_id)
        dst_idx = self._get_or_assign_idx(target_id)

        # ----- Build tensors -------
        # Parse event time (s); Enforce montonic non-decreasing timestamps
        t_val = self._parse_ts(t_iso)
        last_t = getattr(self, "_last_t", t_val)
        if t_val < last_t:
            t_val = last_t
        self._last_t = t_val

        t_tensor = torch.tensor([t_val], dtype=torch.float32, device=self.device)
        src_tensor = torch.tensor([src_idx], dtype=torch.long, device=self.device)
        dst_tensor = torch.tensor([dst_idx], dtype=torch.long, device=self.device)

        # Edge features -> 1D tensor = edge_dim
        e_attr = self._edge_attr_from_features(event.get("features"))
        expected_dim = self.edge_dim

        if e_attr is None:
            e_attr = torch.zeros(expected_dim, dtype=torch.float32, device=self.device)
        else: 
            if not torch.is_tensor(e_attr):
                e_attr = torch.as_tensor(e_attr, dtype=torch.float32, device=self.device)
            else:
                e_attr = e_attr.flatten()[: expected_dim]

        # pad/trim to expected_dim
        flat = e_attr.flatten()
        if flat.numel() < expected_dim:
            pad = torch.zeros(expected_dim - flat.numel(), dtype=torch.float32, device=self.device)
            e_attr = torch.cat([flat, pad], dim=0)
        else:
            e_attr = flat[:expected_dim]

        e_attr_batched = e_attr.view(1, -1)  # (1, edge_dim)

        # ----- INference: detach. forward, update -----
        self.model.eval()

        # Measure detach cost
        up_start = time.perf_counter()
        try:
            self.model.memory.detach() # cut historical DAG in memory before readin git
        except Exception:
            pass

        update_ms = (time.perf_counter() - up_start) * 1000.0

        # sanity checks
        # NOTE: remove if noisy
        assert src_tensor.dim() == 1 and dst_tensor.dim() == 1 and t_tensor.dim() == 1
        assert e_attr_batched.dim() == 2 and e_attr_batched.size(1) == self.edge_dim

        # Forward pass (model returns shape [B, 1])
        fwd_start = time.perf_counter()
        with torch.no_grad():
            # Update memory with new event
            t_event = t_tensor.long()
            self.model.memory.update_state(src_tensor, dst_tensor, t_event, e_attr_batched)

            # Forward pass to get growth score
            score_tensor = self.model(src_tensor, dst_tensor, t_tensor, e_attr_batched)

        forward_ms = (time.perf_counter() - fwd_start) * 1000.0

        # Log latencies (append-only, non-blocking)
        self._log_latencies(update_ms, forward_ms)

        # --- Single growth signal ----
        growth_raw = float(score_tensor.squeeze().item())
        # Optional EMA normalisation (keeps scale stable over time)
        self._ema_growth_mu, self._ema_growth_var = self._ema_update(
            growth_raw, self._ema_growth_mu, self._ema_growth_var
        )
        return {
            "growth_score": growth_raw,
            "score": growth_raw,  # alias for dashboards expecting 'score'
        }
