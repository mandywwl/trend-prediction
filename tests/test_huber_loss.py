import math

import torch

from config.config import HUBER_DELTA_DEFAULT
from models.losses import huber_loss, HuberLoss


def test_huber_elementwise_piecewise_matches_definition():
    delta = 1.5
    pred = torch.tensor([-3.0, -1.5, -0.1, 0.0, 0.2, 1.6, 4.0])
    target = torch.zeros_like(pred)
    loss = huber_loss(pred, target, delta=delta)

    # Manual piecewise
    out = []
    for r in pred.tolist():
        ar = abs(r)
        if ar <= delta:
            out.append(0.5 * r * r)
        else:
            out.append(delta * (ar - 0.5 * delta))
    expected = torch.tensor(out)

    assert torch.allclose(loss, expected, atol=1e-6)


def test_huber_smooth_at_zero_and_boundary():
    delta = 0.7
    # At r = 0 => loss == 0
    l0 = huber_loss(torch.tensor([0.0]), torch.tensor([0.0]), delta=delta)
    assert l0.item() == 0.0

    # At boundary r = +/- delta -> both branches equal
    for sign in (-1.0, 1.0):
        r = torch.tensor([sign * delta])
        l = huber_loss(r, torch.tensor([0.0]), delta=delta).item()
        quad = 0.5 * (delta ** 2)
        lin = delta * (abs(delta) - 0.5 * delta)
        assert math.isclose(l, quad, rel_tol=0, abs_tol=1e-8)
        assert math.isclose(l, lin, rel_tol=0, abs_tol=1e-8)


def test_huber_module_reduction_and_default_delta():
    # Default uses config HUBER_DELTA_DEFAULT
    pred = torch.tensor([0.0, 2.0 * HUBER_DELTA_DEFAULT])
    target = torch.zeros_like(pred)

    m_mean = HuberLoss()
    m_none = HuberLoss(reduction="none")

    mean_val = m_mean(pred, target)
    none_val = m_none(pred, target)

    # Reduction correctness
    assert torch.isclose(mean_val, none_val.mean())

