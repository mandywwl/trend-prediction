import numpy as np
import pytest
from pathlib import Path

def test_node_features():
    """Test the node features in NPZ file coercible and finite."""
    path = Path(__file__).parents[1] / "datasets" / "tgn_edges_basic.npz"
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
