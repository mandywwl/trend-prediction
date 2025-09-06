"""Growth Per Post (GPP) heuristic baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class GPPHeuristic:
    """Compute growth per post over a sliding window.

    The heuristic predicts future growth as the relative increase in counts
    between the most recent value and the value ``window`` steps ago.
    """

    window: int = 1

    def predict(self, counts: Sequence[float]) -> float:
        if len(counts) <= self.window:
            return 0.0
        past = float(counts[-self.window - 1])
        curr = float(counts[-1])
        growth = curr - past
        return growth / max(1.0, past)
