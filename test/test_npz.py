import numpy as np
import pytest
from pathlib import Path

def test_npz():
    """Test the NPZ file basic structure and contents."""
    path = Path(__file__).parents[1] / "data" / "tgn_edges_basic.npz"
    if not path.exists():
        pytest.skip(f"Missing test data: {path}")

    data = np.load(str(path), allow_pickle=True)

    # Must have keys
    required = {"src", "dst", "t", "edge_attr", "node_map", "node_features"}
    assert required.issubset(set(data.files)), f"Keys found: {sorted(data.files)}"

    # Quick sanity checks
    src = np.asarray(data["src"])
    dst = np.asarray(data["dst"])
    t = np.asarray(data["t"])
    edge_attr = np.asarray(data["edge_attr"])

    assert src.ndim == dst.ndim == t.ndim == 1
    assert src.size == dst.size == t.size
    assert edge_attr.shape[0] == src.size


# data_file = Path(__file__).resolve().parent.parent / 'data' / 'tgn_edges_basic.npz'
# print("Loading:", data_file)
# data = np.load(str(data_file), allow_pickle=True)

# # print("src:", data['src'])
# # print("dst:", data['dst'])
# # print("t:", data['t'])
# # print("edge_attr:", data['edge_attr'])
# # print("node_map:", data['node_map']) 

# # Print available keys
# print("keys:", list(data.keys()))

# # Inspect common fields with shape/dtype info
# for k in ['src', 'dst', 't', 'edge_attr', 'node_map', 'node_features']:
#     if k in data:
#         v = data[k]
#         try:
#             shape = np.shape(v)
#             dtype = getattr(v, 'dtype', type(v))
#             print(f"{k}: shape={shape}, dtype={dtype}")
#             # Print content if small
#             try:
#                 size = np.size(v)
#             except Exception:
#                 size = None
#             if size is not None and size <= 20:
#                 print(f"{k} content:", v)
#         except Exception as e:
#             print(f"{k}: cannot inspect ({e})")
#     else:
#         print(f"{k}: not found")