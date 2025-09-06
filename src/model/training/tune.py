"""Hyperparameter tuning for TGN model using Optuna.

This module defines an Optuna objective that performs rolling
week-based cross-validation over the temporal edge dataset.  Each
trial samples hyperparameters from the search space defined in
section \u00a73.7 of the specification and logs the evaluated
configurations.  The best configuration is written to
``tuning_log.json`` in this directory.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Generator, Iterable, Tuple

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    import optuna
except Exception:  # pragma: no cover - keep import-time errors lazily
    optuna = None  # type: ignore

from model.core.tgn import TGNModel


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data or fall back to a tiny synthetic dataset.

    The repository ships without large datasets so this helper generates a
    deterministic synthetic one when the expected ``.npz`` file is absent.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(project_root, "datasets", "tgn_edges_basic.npz")
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        return (
            data["src"],
            data["dst"],
            data["t"],
            data["edge_attr"],
            data["node_features"],
        )

    # Fallback synthetic dataset spanning several weeks.
    rng = np.random.default_rng(0)
    src = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    dst = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    week = 60 * 60 * 24 * 7
    t = np.array([i * week for i in range(len(src))], dtype=float)
    edge_attr = rng.normal(size=(len(src), 5)).astype(np.float32)
    node_feats = rng.normal(size=(3, 5)).astype(np.float32)
    return src, dst, t, edge_attr, node_feats


# Load data once so trials reuse it without reloading from disk.
SRC_ARR, DST_ARR, T_ARR, EDGE_ATTR_ARR, NODE_FEATS = _load_data()
NUM_NODES = NODE_FEATS.shape[0]
EDGE_DIM = EDGE_ATTR_ARR.shape[1]


def _week_id(ts: float) -> int:
    dt = datetime.utcfromtimestamp(float(ts))
    iso = dt.isocalendar()
    return iso[0] * 100 + iso[1]  # combine year and week


def rolling_week_splits(timestamps: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield rolling train/validation indices based on ISO week numbers."""

    week_ids = np.array([_week_id(t) for t in timestamps])
    unique_weeks = np.sort(np.unique(week_ids))
    for i in range(len(unique_weeks) - 1):
        train_weeks = unique_weeks[: i + 1]
        val_week = unique_weeks[i + 1]
        train_idx = np.where(np.isin(week_ids, train_weeks))[0]
        val_idx = np.where(week_ids == val_week)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


def _train_and_eval(params: dict[str, Any], train_idx: np.ndarray, val_idx: np.ndarray) -> float:
    """Train a TGNModel with ``params`` and return validation loss."""

    model = TGNModel(
        num_nodes=NUM_NODES,
        node_feat_dim=NODE_FEATS.shape[1],
        edge_feat_dim=EDGE_DIM,
        time_dim=params["time_dim"],
        memory_dim=params["memory_dim"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()

    src = torch.LongTensor(SRC_ARR)
    dst = torch.LongTensor(DST_ARR)
    t = torch.FloatTensor(T_ARR)
    edge_attr = torch.FloatTensor(EDGE_ATTR_ARR)

    model.train()
    model.reset_memory()
    for i in train_idx:
        src_i = src[i].unsqueeze(0)
        dst_i = dst[i].unsqueeze(0)
        t_i = t[i].unsqueeze(0)
        edge_feat = edge_attr[i].unsqueeze(0)
        label = torch.tensor([1.0])  # placeholder label

        out = model(src_i, dst_i, t_i, edge_feat)
        loss = criterion(out.view(-1), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.memory.detach()
        model.memory.update_state(src_i, dst_i, t_i.long(), edge_feat)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in val_idx:
            src_i = src[i].unsqueeze(0)
            dst_i = dst[i].unsqueeze(0)
            t_i = t[i].unsqueeze(0)
            edge_feat = edge_attr[i].unsqueeze(0)
            label = torch.tensor([1.0])

            out = model(src_i, dst_i, t_i, edge_feat)
            loss = criterion(out.view(-1), label)
            val_loss += loss.item()

            model.memory.detach()
            model.memory.update_state(src_i, dst_i, t_i.long(), edge_feat)

    return val_loss / max(len(val_idx), 1)


def objective(trial: "optuna.trial.Trial") -> float:
    """Optuna objective with the search space from \u00a73.7."""

    params = {
        # Search space replicating section \u00a73.7
        "memory_dim": trial.suggest_int("memory_dim", 64, 256, step=64),
        "time_dim": trial.suggest_int("time_dim", 4, 32, step=4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }
    logger.info("Trial %d params: %s", trial.number, params)

    scores = [
        _train_and_eval(params, train_idx, val_idx)
        for train_idx, val_idx in rolling_week_splits(T_ARR)
    ]

    return float(np.mean(scores)) if scores else float("inf")


def tune_hyperparameters(n_trials: int = 20) -> Any:
    """Run the Optuna study and log the best configuration."""

    if optuna is None:  # pragma: no cover - fail fast when dependency missing
        raise ImportError("Optuna is required for hyperparameter tuning")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best params: %s", study.best_params)

    log_path = os.path.join(os.path.dirname(__file__), "tuning_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f, indent=2)

    return study


if __name__ == "__main__":  # pragma: no cover
    tune_hyperparameters()

