import torch
import torch.nn.functional as F

from config.config import LABEL_SMOOTH_EPS
from model.training.train import smooth_labels, smoothed_cross_entropy


def test_smooth_labels_uses_epsilon():
    target = torch.tensor([1])
    smoothed = smooth_labels(target, num_classes=2)
    expected = torch.tensor([[LABEL_SMOOTH_EPS / 2, 1 - LABEL_SMOOTH_EPS + LABEL_SMOOTH_EPS / 2]])
    assert torch.allclose(smoothed, expected, atol=1e-6)


def test_smoothed_cross_entropy_matches_manual():
    logits = torch.tensor([[0.2, 0.8, -0.5]])
    target = torch.tensor([1])
    eps = 0.1
    loss = smoothed_cross_entropy(logits, target, num_classes=3, eps=eps)

    one_hot = torch.tensor([[0.0, 1.0, 0.0]])
    smoothed = (1 - eps) * one_hot + eps / 3
    log_probs = F.log_softmax(logits, dim=-1)
    expected = -(smoothed * log_probs).sum(dim=-1).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
