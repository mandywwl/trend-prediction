from typing import Optional, Literal

import torch
from torch import Tensor, nn

from config.config import HUBER_DELTA_DEFAULT


def huber_loss(pred: Tensor, target: Tensor, delta: float | None = None) -> Tensor:
    """Element-wise Huber loss between ``pred`` and ``target``.

    Uses standard Huber definition (a.k.a. smooth L1) with piecewise form:

        if |r| <= δ: 0.5 * r^2
        else:        δ * (|r| - 0.5 * δ)

    where r = pred - target and δ defaults to ``HUBER_DELTA_DEFAULT`` from config.

    Args:
        pred: Predictions tensor.
        target: Targets tensor (broadcastable to ``pred``).
        delta: Optional δ. When ``None``, uses config default.

    Returns:
        Tensor of element-wise losses (same broadcasted shape as inputs).
    """
    if delta is None:
        delta = float(HUBER_DELTA_DEFAULT)

    pred, target = torch.as_tensor(pred), torch.as_tensor(target)
    r = pred - target
    abs_r = r.abs()
    delta_t = torch.tensor(delta, dtype=abs_r.dtype, device=abs_r.device)

    quadratic = 0.5 * r * r
    linear = delta_t * (abs_r - 0.5 * delta_t)
    return torch.where(abs_r <= delta_t, quadratic, linear)


class HuberLoss(nn.Module):
    """Huber loss as an ``nn.Module``.

    Defaults ``delta`` to ``HUBER_DELTA_DEFAULT`` when not provided.
    """

    def __init__(
        self,
        delta: Optional[float] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = huber_loss(pred, target, delta=self.delta)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def log_huber_vs_mse_curve(
    *,
    path: str,
    residual_min: float = -5.0,
    residual_max: float = 5.0,
    steps: int = 201,
    delta: float | None = None,
) -> None:
    """Write a CSV comparing Huber and MSE across residuals.

    Useful for quick ablation plots without additional dependencies.
    Creates or overwrites ``path``.
    """
    import os

    if delta is None:
        delta = float(HUBER_DELTA_DEFAULT)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    rs = torch.linspace(residual_min, residual_max, steps)
    zeros = torch.zeros_like(rs)
    hub = huber_loss(rs, zeros, delta=delta)
    mse = 0.5 * rs * rs  # comparable 0.5 * r^2 scaling

    with open(path, "w", encoding="utf-8") as f:
        f.write("residual,huber,mse\n")
        for r, h, m in zip(rs.tolist(), hub.tolist(), mse.tolist()):
            f.write(f"{r},{h},{m}\n")
