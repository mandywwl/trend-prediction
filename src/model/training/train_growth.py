""" Optional standalone training/ablation for a Growth-only head that uses Huber;

Delta defaults to HUBER_DELTA_DEFAULT from config, and can be overridden
via YAML (if provided). Also logs Huber vs MSE loss curves for ablations.
"""

import os
from typing import Optional

import torch

from config.config import HUBER_DELTA_DEFAULT
from model.core.losses import HuberLoss, log_huber_vs_mse_curve
from utils.io import maybe_load_yaml
from utils.path_utils import find_repo_root


def train_growth(yaml_path: Optional[str] = None) -> None:
    cfg = maybe_load_yaml(yaml_path)
    huber_cfg = cfg.get("huber", {}) if isinstance(cfg, dict) else {}
    delta = huber_cfg.get("delta", None)

    # Model stub: y_hat = Wx + b for demonstration; replace with real model
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 1),
    )

    # Use robust Huber loss; falls back to config default if delta is None
    criterion = HuberLoss(delta=delta, reduction="mean")
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic regression data with 10% outliers to validate stability
    torch.manual_seed(0)
    n = 2048
    X = torch.randn(n, 16)
    true_w = torch.randn(16, 1)
    y = X @ true_w + 0.1 * torch.randn(n, 1)

    # Inject 10% large noise outliers
    idx = torch.randperm(n)[: int(0.10 * n)]
    y[idx] += 10.0 * torch.sign(torch.randn_like(y[idx]))

    for epoch in range(5):
        optim.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optim.step()

    # Log Huber vs MSE curve for ablations
    out_path = find_repo_root() / "datasets" / "huber_vs_mse.csv"
    log_huber_vs_mse_curve(
        path=out_path, delta=delta if delta is not None else HUBER_DELTA_DEFAULT
    )


if __name__ == "__main__":
    # Optional: pass YAML path via TRAIN_GROWTH_CFG env var
    train_growth(os.environ.get("TRAIN_GROWTH_CFG"))
