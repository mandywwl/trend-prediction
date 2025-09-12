"""Train a Growth-only regressor on future labels (velocity) produced in preprocessing.

Also supports optional acceleration target and logs Huber vs MSE curves for ablations.
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from config.config import (
    DATA_DIR,
    HUBER_DELTA_DEFAULT,
    INFERENCE_DEVICE,
    # Optional knobs (add these to config.py if not present; defaults used otherwise)
    # GROWTH_OBJECTIVE: "velocity" (use y_growth) or "accel" (difference of consecutive y_growth per topic)
)
from model.core.losses import HuberLoss, log_huber_vs_mse_curve
from utils.io import maybe_load_yaml, ensure_dir
from utils.path_utils import find_repo_root

def _standardise(y: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    mean = float(y.mean().item())
    std = float(y.std(unbiased=False).item() or 1.0)
    yz = (y - mean) / std
    return yz, mean, std

def _compute_accel_from_edge_bins(y_growth: np.ndarray,
                                  topic_ids: np.ndarray,
                                  time_bins: np.ndarray) -> np.ndarray:
    """Acceleration = Δ y_growth across consecutive bins per topic, aligned back to each edge sample.
    For the first occurrence of a (topic, time_bin) we set accel = 0.0.
    """
    order = np.lexsort((time_bins, topic_ids))
    t_sorted = topic_ids[order]
    y_sorted = y_growth[order]

    accel_sorted = np.zeros_like(y_sorted, dtype=np.float32)
    i = 0
    n = len(y_sorted)
    while i < n:
        j = i
        while j < n and t_sorted[j] == t_sorted[i]:
            j += 1
        seg = slice(i, j)
        y_seg = y_sorted[seg]
        dy = np.diff(y_seg, prepend=y_seg[0:1])
        accel_sorted[seg] = dy.astype(np.float32)
        i = j

    accel = np.zeros_like(y_growth, dtype=np.float32)
    accel[order] = accel_sorted
    return accel

def _load_npz(path: Optional[str] = None):
    npz_path = path or (DATA_DIR / "tgn_edges_basic.npz")
    data = np.load(npz_path, allow_pickle=True)
    X = data["edge_attr"].astype(np.float32)
    y_vel = data["y_growth"].astype(np.float32)
    topic_ids = data["edge_topic_ids"].astype(np.int64)
    time_bins = data["edge_time_bins"].astype(np.int64)
    return X, y_vel, topic_ids, time_bins


def train_growth(yaml_path: Optional[str] = None) -> None:
    # ------ Config / IO -------
    cfg = maybe_load_yaml(yaml_path)
    huber_cfg = cfg.get("huber", {}) if isinstance(cfg, dict) else {}
    delta = huber_cfg.get("delta", None)

    obj = (cfg.get("growth_objective") if isinstance(cfg, dict) else None) or (globals().get("GROWTH_OBJECTIVE") or "velocity")
    npz_override = cfg.get("npz_path") if isinstance(cfg, dict) else None

    device = torch.device(INFERENCE_DEVICE if INFERENCE_DEVICE else "cpu")
    out_dir = DATA_DIR
    ensure_dir(out_dir)
    ckpt_path = out_dir / ("growth_head.pt" if obj == "velocity" else "accel_head.pt")

    # ------ Data loading -------
    X_np, y_vel_np, topic_ids, time_bins = _load_npz(npz_override)
    if obj == "accel":
        y_np = _compute_accel_from_edge_bins(y_vel_np, topic_ids, time_bins)
    else:
        y_np = y_vel_np

    X = torch.from_numpy(X_np).to(device)  # (N, D)
    y = torch.from_numpy(y_np).unsqueeze(1).to(device) # (N, 1)

    y_std, y_mean, y_scale = _standardise(y)

    ds = TensorDataset(X, y_std)
    # simple 90/10 split
    n = len(ds)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=2048, shuffle=False, drop_last=False)

    # -------- Model  -------
    in_dim = X.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, max(64, in_dim // 4)),
        torch.nn.ReLU(),
        torch.nn.Linear(max(64, in_dim // 4), 1),
    ).to(device)

    criterion = HuberLoss(delta=delta, reduction="mean").to(device) # uses default if delta is None
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------- Training loop -------
    def _epoch(split: str):
        dl = train_dl if split == "train" else val_dl
        total, count = 0.0, 0
        if split == "train":
            model.train()
        else:
            model.eval()
        with torch.set_grad_enabled(split == "train"):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                if split == "train":
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total += float(loss.item()) * yb.size(0)
                count += yb.size(0)
        return total / max(1, count)
    
    epochs = int(cfg.get("epochs") if isinstance(cfg, dict) else None) or (globals().get("TRAIN_EPOCHS") or 8)

    for e in range(epochs):
        tr = _epoch("train")
        va = _epoch("val")
        print(f"Epoch {e} | train_huber: {tr:.4f} | val_huber: {va:.4f}")

    # ------- Save model -------
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": in_dim,
            "objective": obj,
            "label_mean": y_mean,
            "label_std": y_scale,
            "delta": delta if delta is not None else HUBER_DELTA_DEFAULT,
            "npz_used": str((cfg or {}).get("npz_path") or DATA_DIR / "tgn_edges_basic.npz"),
        },
        ckpt_path,
    )
    print(f"[train_growth] Saved growth head model to {ckpt_path}")


    # --------- Ablation logging -------
    out_path = DATA_DIR/ "huber_vs_mse.csv"
    log_huber_vs_mse_curve(
        path=out_path, delta=delta if delta is not None else HUBER_DELTA_DEFAULT
    )
    print(f"[train_growth] Logged Huber vs MSE curve to {out_path}")


if __name__ == "__main__":
    # Optional: pass YAML path via TRAIN_GROWTH_CFG env var
    train_growth(os.environ.get("TRAIN_GROWTH_CFG"))
