import torch
import torch.nn as nn
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    TimeEncoder,
)


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

        # Decoder now produces three values per edge interaction:
        #   1. emergence probability
        #   2. growth rate
        #   3. diffusion score
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim * 2 + time_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 3),
        )

    def forward(self, src, dst, t, edge_attr):
        # Get last update time for src node(s)
        src_last_update = self.memory.last_update[src]
        delta_t = t - src_last_update
        delta_t_enc = self.time_encoder(delta_t)

        # Memory lookup
        src_mem = self.memory.memory[src]
        dst_mem = self.memory.memory[dst]

        # Decode
        out = torch.cat([src_mem, dst_mem, delta_t_enc], dim=1)
        return self.decoder(out)

    def reset_memory(self):
        self.memory.reset_state()
