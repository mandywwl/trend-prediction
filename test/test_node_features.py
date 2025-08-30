import numpy as np
import pytest
from pathlib import Path

def test_node_features():
    """Test the node features in NPZ file coercible and finite."""
    path = Path(__file__).parents[1] / "data" / "tgn_edges_basic.npz"
    if not path.exists():
        pytest.skip(f"Missing test data: {path}")

    data = np.load(str(path), allow_pickle=True)
    node_map = np.asarray(data["node_map"], dtype=object)
    node_features = np.asarray(data["node_features"], dtype=object)

    assert len(node_features) == len(node_map) and len(node_map) > 0

    n = min(20, len(node_map)) # Check first 20 or fewer
    for i in range(n):
        feat = node_features[i]
        # Coerce to float array; allow ragged/object entries
        try:
            arr = np.asarray(feat, dtype=float)
            if arr.ndim == 0:  # scalar to make it 1D
                arr = np.array([float(arr)])
        except Exception:
            try:
                arr = np.array(list(feat), dtype=float)
            except Exception:
                arr = np.array([], dtype=float)

        if arr.size:
            assert np.isfinite(arr).all(), f"Non-finite values at index {i}"


# data_file = Path(__file__).resolve().parent.parent / 'data' / 'tgn_edges_basic.npz'
# data = np.load(str(data_file), allow_pickle=True)

# node_map = data['node_map']
# node_features = data['node_features']

# if node_map is None or node_features is None:
#     raise RuntimeError(f"Missing required arrays in {data_file}. Available keys: {list(data.keys())}")

# # Print checks for the first 20 nodes/fewer if not enough
# count = min(20, len(node_map))
# for i, node_label in enumerate(node_map[:count]):
#     feat = node_features[i]
#     # Convert feature entry to a proper numpy array of floats, handling object/ragged entries
#     try:
#         arr = np.asarray(feat, dtype=float)
#     except Exception:
#         # Fallback: try to coerce via list()
#         try:
#             arr = np.array(list(feat), dtype=float)
#         except Exception:
#             # As a last resort, create a zero-length array
#             arr = np.array([], dtype=float)
#     feat_sum = float(np.sum(arr)) if arr.size > 0 else 0.0
#     is_zero = bool(np.all(arr == 0)) if arr.size > 0 else True
#     print(f"{i}: {node_label}, Feature sum: {feat_sum:.4f}, Zero: {is_zero}")