import torch
import torch.nn as nn
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    TimeEncoder,
)
from config.config import TGN_DECODER_HIDDEN

class TGNModel(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, edge_feat_dim, time_dim, memory_dim):
        super().__init__()

        self.time_encoder = TimeEncoder(time_dim)
        self.message_module = IdentityMessage(edge_feat_dim, memory_dim, time_dim)
        self.aggregator_module = LastAggregator()

        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=edge_feat_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=self.message_module,
            aggregator_module=self.aggregator_module,
        )


        # Single-neuron regression head -> predicted growth score (scaler)
        self.in_dim = memory_dim * 2 + time_dim + edge_feat_dim
        self.growth_head = nn.Sequential(
            nn.Linear(self.in_dim, TGN_DECODER_HIDDEN),
            nn.ReLU(),
            nn.Linear(TGN_DECODER_HIDDEN, 1),
        )

    def forward(self, src, dst, t, edge_attr):
        # Get last update time for src node(s)
        src_last_update = self.memory.last_update[src]
        delta_t = t - src_last_update
        delta_t_enc = self.time_encoder(delta_t)

        # Memory lookup
        src_mem = self.memory.memory[src]
        dst_mem = self.memory.memory[dst]

        # ENsure edge_attr is 2D and o same device/dtype
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(0)
        edge_attr = edge_attr.to(src_mem.device).float()

        # Decode
        out = torch.cat([src_mem, dst_mem, delta_t_enc, edge_attr], dim=1)
        growth_score = self.growth_head(out) # (N, 1)
        return growth_score

    def reset_memory(self):
        self.memory.reset_state()
