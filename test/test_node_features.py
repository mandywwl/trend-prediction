import numpy as np

data = np.load('../data/tgn_edges_basic.npz', allow_pickle=True)
node_map = data['node_map']
node_features = data['node_features']

for i, node_label in enumerate(node_map[:20]):  # first 20 nodes, increase if needed
    feat_sum = node_features[i].sum()
    is_zero = np.all(node_features[i] == 0)
    print(f"{i}: {node_label}, Feature sum: {feat_sum:.4f}, Zero: {is_zero}")
