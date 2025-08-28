import numpy as np
from pathlib import Path

# Resolve data file relative to this test file
data_file = Path(__file__).resolve().parent.parent / 'data' / 'tgn_edges_basic.npz'
print("Loading:", data_file)
data = np.load(str(data_file), allow_pickle=True)

# print("src:", data['src'])
# print("dst:", data['dst'])
# print("t:", data['t'])
# print("edge_attr:", data['edge_attr'])
# print("node_map:", data['node_map']) 

# Print available keys
print("keys:", list(data.keys()))

# Inspect common fields with shape/dtype info
for k in ['src', 'dst', 't', 'edge_attr', 'node_map', 'node_features']:
    if k in data:
        v = data[k]
        try:
            shape = np.shape(v)
            dtype = getattr(v, 'dtype', type(v))
            print(f"{k}: shape={shape}, dtype={dtype}")
            # Print content if small
            try:
                size = np.size(v)
            except Exception:
                size = None
            if size is not None and size <= 20:
                print(f"{k} content:", v)
        except Exception as e:
            print(f"{k}: cannot inspect ({e})")
    else:
        print(f"{k}: not found")