import numpy as np
from pathlib import Path

# Resolve data file relative to this test file
data_file = Path(__file__).resolve().parent.parent / 'data' / 'tgn_edges_basic.npz'
data = np.load(str(data_file), allow_pickle=True)

node_map = data['node_map']
node_features = data['node_features']

if node_map is None or node_features is None:
    raise RuntimeError(f"Missing required arrays in {data_file}. Available keys: {list(data.keys())}")

# Print checks for the first 20 nodes/fewer if not enough
count = min(20, len(node_map))
for i, node_label in enumerate(node_map[:count]):
    feat = node_features[i]
    # Convert feature entry to a proper numpy array of floats, handling object/ragged entries
    try:
        arr = np.asarray(feat, dtype=float)
    except Exception:
        # Fallback: try to coerce via list()
        try:
            arr = np.array(list(feat), dtype=float)
        except Exception:
            # As a last resort, create a zero-length array
            arr = np.array([], dtype=float)
    feat_sum = float(np.sum(arr)) if arr.size > 0 else 0.0
    is_zero = bool(np.all(arr == 0)) if arr.size > 0 else True
    print(f"{i}: {node_label}, Feature sum: {feat_sum:.4f}, Zero: {is_zero}")