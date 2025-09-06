"""Simple snapshot LSTM baseline implemented with NumPy.

This baseline models a sequence of scalar features with a single-layer LSTM
and returns the final hidden state as the prediction. It is intentionally
lightweight and does not rely on PyTorch so that it can run in environments
where heavy dependencies are unavailable. The implementation is minimal but
captures the essential gating behaviour of an LSTM cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SnapshotLSTM:
    """A single-layer LSTM operating on snapshots.

    Parameters
    ----------
    input_size: int
        Number of input features per timestep.
    hidden_size: int
        Size of the hidden state.
    output_size: int, optional
        Size of the output. Defaults to ``1``.
    """

    input_size: int
    hidden_size: int
    output_size: int = 1

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        in_dim = self.input_size + self.hidden_size
        # Concatenated weight matrix for i, f, g, o gates
        self.W = rng.standard_normal((in_dim, 4 * self.hidden_size)) * 0.1
        self.b = np.zeros(4 * self.hidden_size)
        self.W_out = rng.standard_normal((self.hidden_size, self.output_size)) * 0.1
        self.b_out = np.zeros(self.output_size)

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass on a batch of sequences.

        Parameters
        ----------
        x: ndarray of shape ``(batch, seq_len, input_size)``

        Returns
        -------
        ndarray of shape ``(batch, output_size)``
        """

        batch, seq_len, feat = x.shape
        if feat != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got {feat}")

        h = np.zeros((batch, self.hidden_size))
        c = np.zeros((batch, self.hidden_size))
        for t in range(seq_len):
            inp = np.concatenate([x[:, t, :], h], axis=1)
            gates = inp @ self.W + self.b
            i, f, g, o = np.split(gates, 4, axis=1)
            i = _sigmoid(i)
            f = _sigmoid(f)
            g = np.tanh(g)
            o = _sigmoid(o)
            c = f * c + i * g
            h = o * np.tanh(c)
        out = h @ self.W_out + self.b_out
        return out.squeeze(-1)

    # ------------------------------------------------------------------
    def predict(self, sequence: list[float] | np.ndarray) -> float:
        """Convenience prediction for a single sequence of scalars."""
        arr = np.asarray(sequence, dtype=float).reshape(1, -1, self.input_size)
        return float(self.forward(arr)[0])
