import numpy as np

from model.baselines import SnapshotLSTM, GPPHeuristic
from model.evaluation import evaluate_baselines


def test_snapshot_lstm_forward_shape():
    model = SnapshotLSTM(input_size=1, hidden_size=2)
    x = np.random.randn(3, 4, 1)
    out = model.forward(x)
    assert out.shape == (3,)


def test_gpp_heuristic_growth():
    gpp = GPPHeuristic(window=1)
    score = gpp.predict([10, 15, 25])
    assert abs(score - (25 - 15) / 15) < 1e-6


def test_baseline_evaluation_integration():
    sequences = {1: [1, 2, 3], 2: [2, 3, 4]}
    ts = "2025-01-01T00:00:00"
    results = evaluate_baselines(sequences, emergent_topics=[1], ts_iso=ts)
    for snap in results.values():
        assert set(["k5", "k10", "support"]) <= snap.keys()
        assert snap["support"] == 1
