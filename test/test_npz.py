import numpy as np

data = np.load('../data/tgn_edges_basic.npz', allow_pickle=True)
print("src:", data['src'])
print("dst:", data['dst'])
print("t:", data['t'])
print("edge_attr:", data['edge_attr'])
print("node_map:", data['node_map']) 
