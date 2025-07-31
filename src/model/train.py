import numpy as np
import os
import torch
from model.tgn import TGNModel

# Load the preprocessed data
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # abspath to the project root
data_dir = os.path.join(project_root, 'data/tgn_edges_basic.npz')
data = np.load(data_dir, allow_pickle=True)

# Extract data from the npz file
src = torch.LongTensor(data['src'])
dst = torch.LongTensor(data['dst'])
t = torch.FloatTensor(data['t'])
edge_attr = torch.FloatTensor(data['edge_attr'])
node_feats = torch.FloatTensor(data['node_features'])
num_nodes = node_feats.shape[0]

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

for epoch in range(3):  # TODO: Example, 3 epochs for quick testing
    model.reset_memory()  # Reset memory state at the start of each epoch
    total_loss = 0.0


    for i in range(len(src) - 1): # NOTE: No negative sampling yet, just for testing
    
        src_i = src[i].unsqueeze(0).long()
        dst_i = dst[i].unsqueeze(0).long()
        t_i = t[i].unsqueeze(0)
        edge_feat = edge_attr[i].unsqueeze(0)
        label = torch.tensor([1.0])  # Dummy, TODO: add negatives

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