import logging
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F

from model.core.tgn import TGNModel
from model.training.noise_injection import inject_noise
from model.core.losses import HuberLoss
from config.schemas import Batch
from config.config import (
    DATA_DIR,
    LABEL_SMOOTH_EPS,
    NOISE_P_DEFAULT,
    TGN_MEMORY_DIM,
    TGN_TIME_DIM,
    TGN_LR,
    INFERENCE_DEVICE,
    HUBER_DELTA_DEFAULT,
    TRAIN_EPOCHS,
)
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def train_tgn_from_npz(npz_path: str, ckpt_out: str, 
                       epochs: int | None = None, 
                       device: str = INFERENCE_DEVICE,
                       noise_p: float | None = None,
                       noise_seed: int | None = None
                       ) -> None:
    """ Train a TGN model from an NPZ dataset and save the checkpoint.
    Args:
        npz_path: Path to the NPZ file containing the dataset.
        ckpt_out: Path to save the trained model checkpoint.
        epochs: Number of training epochs. If None, falls back to config.TRAIN_EPOCHS.
        device: Device to run the training on ("cpu" or "cuda").
    
    """
    # Resolve epochs from arg or config
    epochs = int(epochs) if epochs is not None else int(TRAIN_EPOCHS)

    # Load data
    data = np.load(npz_path, allow_pickle=True)
    src_arr = data["src"]
    dst_arr = data["dst"]
    t_arr = data["t"]
    edge_attr_arr = data["edge_attr"]
    y_growth_arr = data["y_growth"].astype(np.float32)  # per-edge growth label
    y_growth_arr = np.nan_to_num(y_growth_arr, nan=0.0, posinf=0.0, neginf=0.0) # clean up
    
    # node_features are optional; fall back to 0-dim node features
    try:
        node_feats = torch.as_tensor(data["node_features"])
        node_feat_dim = int(node_feats.shape[1]) if node_feats.ndim == 2 else 0
        num_nodes = int(node_feats.shape[0])
    except KeyError:
        # derive num_nodes from edges
        num_nodes = int(max(src_arr.max(), dst_arr.max())) + 1 if len(src_arr) else 1
        node_feat_dim = 0
        node_feats = None

    EDGE_DIM = edge_attr_arr.shape[1]

    events: Batch = []
    for i in range(len(src_arr)):
        events.append(
            {
                "event_id": str(i),
                "ts_iso": datetime.utcfromtimestamp(float(t_arr[i])).replace(tzinfo=timezone.utc).isoformat(),
                "actor_id": str(src_arr[i]),
                "target_ids": [str(dst_arr[i])],
                "edge_type": "edge",
                "features": {"text_emb": edge_attr_arr[i]},
            }
        )

    events = inject_noise(events, p=(noise_p if noise_p is not None else NOISE_P_DEFAULT),
                          seed=noise_seed)
    n = min(len(events), len(y_growth_arr))
    if len(events) != n or len(y_growth_arr) != n:
        logging.warning(
            "length mismatch after noise injection: events=%d, labels=%d → truncating to %d",
            len(events), len(y_growth_arr), n
        )
    events = events[:n]
    y_growth_arr = y_growth_arr[:n]

    dev = torch.device(device)

    src = torch.LongTensor([int(e["actor_id"]) for e in events])
    dst = torch.LongTensor([int(e["target_ids"][0]) for e in events])
    src, dst = src.to(dev), dst.to(dev)
    t = torch.FloatTensor([datetime.fromisoformat(e["ts_iso"]).timestamp() for e in events]).to(dev)

    # edge_attr = torch.from_numpy(
    #     np.stack([e["features"]["text_emb"] for e in events], axis=0)
    # ).to(dev)
    MEM_DIM = TGN_MEMORY_DIM
    TIME_DIM = TGN_TIME_DIM
    EDGE_DIM = edge_attr_arr.shape[1]
    edge_attr_list = []
    missing = 0
    
    for e in events:
        feats = e.get("features") or {}
        emb = feats.get("text_emb")
        if emb is None:
            missing += 1
            # Backfill policy: zeros (you can also support "mean" or "learned_unknown")
            emb = np.zeros(EDGE_DIM, dtype=np.float32)
            feats["text_emb"] = emb           # write back so the event is now normalized
            e["features"] = feats
        edge_attr_list.append(emb)

    edge_attr = torch.from_numpy(
        np.stack(edge_attr_list, dtype=np.float32)
    )
    if missing:
        logging.info("[train.py] backfilled %d missing text_emb", missing)
    
    model = TGNModel(
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=EDGE_DIM,
        time_dim=TIME_DIM,
        memory_dim=MEM_DIM,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=TGN_LR)
    criterion = HuberLoss(delta=HUBER_DELTA_DEFAULT, reduction="mean").to(dev) # robust regression loss for growth
    y = torch.from_numpy(y_growth_arr).to(dev).unsqueeze(1)     # Labels tensor (N, 1)

    # ------- Sanity checks -------
    print("mem_dim,time_dim,edge_dim =>",
          MEM_DIM, TIME_DIM, EDGE_DIM)
    print("edge_attr shape:", tuple(edge_attr.shape))

    # One dry forward to validate shapes
    with torch.no_grad():
        _ = model(src[:1].long(), dst[:1].long(), t[:1], edge_attr[:1].float())
    print("Dry forward OK. Output shape:", tuple(_.shape))

    # ------- Training loop -------
    for epoch in range(epochs):
        model.reset_memory()
        total_loss = 0.0

        for i in range(len(src)):
            src_i = src[i].unsqueeze(0).long().to(dev)
            dst_i = dst[i].unsqueeze(0).long().to(dev)
            t_i = t[i].unsqueeze(0).to(dev)
            edge_feat = edge_attr[i].unsqueeze(0).to(dev)
            # label = torch.tensor([1], device=dev)  # TODO: replace with actual per-edge label

            # # TGN decoder outputs [emergence_logit, growth, diffusion].
            # # For the classification loss we only want a single binary logit.
            # logit_pos = model(src_i, dst_i, t_i, edge_feat)[..., 0:1]   # (B,1)
            # logits = torch.cat([torch.zeros_like(logit_pos), logit_pos], dim=-1)  # (B,2)

            # loss = smoothed_cross_entropy(logits, label, num_classes=2)

            # Regression target (future growth) aligned to edge i
            y_i = y[i].unsqueeze(0)  # (1,1)

            # TGNModel returns a single scalar growth score (B,1)
            pred = model(src_i, dst_i, t_i, edge_feat)  # (1,1)
            loss = criterion(pred, y_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            model.memory.detach()
            t_event = t_i.long()
            model.memory.update_state(src_i, dst_i, t_event, edge_feat)

        print(f"Epoch {epoch} - Loss: {total_loss / (len(src) - 1):.4f}")


    # after training:
    ckpt_payload = {
        "state_dict": model.state_dict(),
        "hparams": {
            "num_nodes": int(num_nodes),
            "memory_dim": int(TGN_MEMORY_DIM),
            "time_dim": int(TGN_TIME_DIM),
            "edge_feat_dim": int(EDGE_DIM),
            "node_feat_dim": int(node_feat_dim),
        }
    }
    ckpt_path = Path(ckpt_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_payload, ckpt_path)
    print(f"[train.py] Saved checkpoint → {ckpt_path}")


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

# NOTE: For quick testing of training loop only.  
if __name__ == "__main__":
    npz_path  = DATA_DIR / "tgn_edges_basic.npz"
    ckpt_out  = DATA_DIR / "tgn_model.pt"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. Run the service to generate events, then preprocess to create it."
        )

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)


    # Use config device
    device = INFERENCE_DEVICE if INFERENCE_DEVICE in ("cpu", "cuda") else "cpu"

    train_tgn_from_npz(str(npz_path), str(ckpt_out), epochs=TRAIN_EPOCHS, device=device)