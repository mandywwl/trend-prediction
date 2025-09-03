import numpy as np
import os
import torch
import logging
from datetime import datetime
from model.tgn import TGNModel
from robustness.noise_injection import inject_noise
from config.schemas import Batch

logging.basicConfig(level=logging.INFO)

# Load the preprocessed data
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_dir = os.path.join(project_root, "data/tgn_edges_basic.npz")
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
t = torch.FloatTensor([datetime.fromisoformat(e["ts_iso"]).timestamp() for e in events])
edge_attr_list = []
for e in events:
    emb = e.get("features", {}).get("text_emb")
    if emb is None:
        emb = np.zeros(EDGE_DIM, dtype=np.float32)
    edge_attr_list.append(emb)
edge_attr = torch.FloatTensor(edge_attr_list)

# # Build edge stream
# edge_stream = []
# for i in range(len(src)):
#     edge_stream.append((
#         src[i].item(),  # Convert to Python int
#         dst[i].item(),  # Convert to Python int
#         t[i].item(),    # Convert to Python float
#         edge_attr[i].numpy()  # Convert to numpy array
#     ))


# ---------- Model ----------
MEM_DIM   = 100
TIME_DIM  = 10
EDGE_DIM  = edge_attr.shape[1]


model = TGNModel(
    num_nodes = num_nodes,
    node_feat_dim = node_feats.shape[1],
    edge_feat_dim = EDGE_DIM,
    time_dim = TIME_DIM,
    memory_dim = MEM_DIM
)


# ------ Training loop ------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(3):  # TODO: 3 epochs for quick testing; increase for actual training
    model.reset_memory()  # Reset memory state at the start of each epoch
    total_loss = 0.0


    for i in range(len(src) - 1): # TODO: Add negative sampling
    
        src_i = src[i].unsqueeze(0).long()
        dst_i = dst[i].unsqueeze(0).long()
        t_i = t[i].unsqueeze(0)
        edge_feat = edge_attr[i].unsqueeze(0)
        label = torch.tensor([1.0]) # Dummy

        # DEBUG
        if i == 0:
            print(src_i.dtype, t_i.dtype, edge_feat.dtype)

        # forward / loss / optimize
        out = model(src_i, dst_i, t_i, edge_feat)
        loss = criterion(out.view(-1), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        model.memory.detach()

        # Update memory
        t_event = t_i.long() # Convert to long for memory update
        model.memory.update_state(src_i, dst_i, t_event, edge_feat)
        
    print(f"Epoch {epoch} - Loss: {total_loss / (len(src)-1):.4f}")
