import os
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

from model.core.tgn import TGNModel
from model.training.noise_injection import inject_noise
from config.schemas import Batch
from config.config import LABEL_SMOOTH_EPS

logging.basicConfig(level=logging.INFO)


def smooth_labels(
    target: torch.Tensor, num_classes: int, eps: float = LABEL_SMOOTH_EPS
) -> torch.Tensor:
    """Apply label smoothing to class indices or one-hot labels.

    Args:
        target: Tensor of class indices or one-hot encoded labels.
        num_classes: Number of classes.
        eps: Smoothing factor ``ε``.

    Returns:
        Smoothed label distribution with shape ``(..., num_classes)``.
    """
    if target.dim() == 1 or target.shape[-1] != num_classes:
        target = F.one_hot(target.long(), num_classes=num_classes).float()
    return (1.0 - eps) * target + eps / num_classes


def smoothed_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = LABEL_SMOOTH_EPS,
) -> torch.Tensor:
    """Cross-entropy with label smoothing.

    Args:
        logits: Logits tensor of shape ``(N, C)``.
        target: Class indices or one-hot labels.
        num_classes: Number of classes ``C``.
        eps: Smoothing factor ``ε``.

    Returns:
        Mean cross-entropy loss with label smoothing applied.
    """
    smoothed = smooth_labels(target, num_classes, eps)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smoothed * log_probs).sum(dim=-1)
    return loss.mean()


if __name__ == "__main__":
    # Example training loop demonstrating usage of smoothed cross-entropy.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "datasets/tgn_edges_basic.npz")
    data = np.load(data_dir, allow_pickle=True)

    src_arr = data["src"]
    dst_arr = data["dst"]
    t_arr = data["t"]
    edge_attr_arr = data["edge_attr"]
    node_feats = torch.FloatTensor(data["node_features"])
    num_nodes = node_feats.shape[0]
    EDGE_DIM = edge_attr_arr.shape[1]

    events: Batch = []
    for i in range(len(src_arr)):
        events.append(
            {
                "event_id": str(i),
                "ts_iso": datetime.utcfromtimestamp(float(t_arr[i])).isoformat(),
                "actor_id": str(src_arr[i]),
                "target_ids": [str(dst_arr[i])],
                "edge_type": "edge",
                "features": {"text_emb": edge_attr_arr[i]},
            }
        )

    events = inject_noise(events, seed=0)

    src = torch.LongTensor([int(e["actor_id"]) for e in events])
    dst = torch.LongTensor([int(e["target_ids"][0]) for e in events])
    t = torch.FloatTensor([
        datetime.fromisoformat(e["ts_iso"]).timestamp() for e in events
    ])
    edge_attr_list = []
    for e in events:
        emb = e.get("features", {}).get("text_emb")
        if emb is None:
            emb = np.zeros(EDGE_DIM, dtype=np.float32)
        edge_attr_list.append(emb)
    edge_attr = torch.FloatTensor(edge_attr_list)

    MEM_DIM = 100
    TIME_DIM = 10
    EDGE_DIM = edge_attr.shape[1]

    model = TGNModel(
        num_nodes=num_nodes,
        node_feat_dim=node_feats.shape[1],
        edge_feat_dim=EDGE_DIM,
        time_dim=TIME_DIM,
        memory_dim=MEM_DIM,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):  # quick testing; increase for actual training
        model.reset_memory()
        total_loss = 0.0

        for i in range(len(src) - 1):
            src_i = src[i].unsqueeze(0).long()
            dst_i = dst[i].unsqueeze(0).long()
            t_i = t[i].unsqueeze(0)
            edge_feat = edge_attr[i].unsqueeze(0)
            label = torch.tensor([1])  # dummy positive class

            logits_single = model(src_i, dst_i, t_i, edge_feat)
            logits = torch.cat([torch.zeros_like(logits_single), logits_single], dim=-1)
            loss = smoothed_cross_entropy(logits, label, num_classes=2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            model.memory.detach()
            t_event = t_i.long()
            model.memory.update_state(src_i, dst_i, t_event, edge_feat)

        print(f"Epoch {epoch} - Loss: {total_loss / (len(src) - 1):.4f}")