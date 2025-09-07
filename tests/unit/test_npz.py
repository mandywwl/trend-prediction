import numpy as np
import pytest

from utils.path_utils import find_repo_root


def test_npz():
    """Test the NPZ file basic structure and contents."""
    path = find_repo_root() / "datasets" / "tgn_edges_basic.npz"
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

