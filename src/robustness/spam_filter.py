from __future__ import annotations

"""Lightweight spam and bot scoring utilities."""

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import math
import numpy as np


@dataclass
class SpamScorerConfig:
    """Configuration for :class:`SpamScorer` heuristics."""

    min_account_age_days: float = 14.0
    min_follower_ratio: float = 0.1
    min_interval_entropy: float = 0.5
    max_non_ascii_ratio: float = 0.3
    blacklist: Sequence[str] = (
        "buy now",
        "free",
        "click here",
        "subscribe",
    )


class SpamScorer:
    """Score accounts and events for spam likelihood.

    The scoring relies on cheap heuristics only so it can run inline with
    streaming inference. Scores are in ``[0, 1]`` where ``0`` denotes a high
    likelihood of the account being synthetic/spam.
    """

    def __init__(self, config: SpamScorerConfig | None = None) -> None:
        self.config = config or SpamScorerConfig()

    # ------------------------------------------------------------------
    def score_account(self, user: Mapping[str, object]) -> float:
        """Return a spam score for ``user`` in ``[0, 1]``.

        The function expects a mapping with keys such as ``created_at`` (ISO
        timestamp), ``followers`` and ``following`` counts and a ``posts``
        sequence where each post contains ``timestamp`` and optional ``text``.
        Missing information defaults to neutral values so the scorer degrades
        gracefully when data is sparse.
        """

        now = datetime.utcnow()
        scores: list[float] = []

        # Account age ----------------------------------------------------
        created = user.get("created_at")
        age_days = 0.0
        if isinstance(created, str):
            created_dt = datetime.fromisoformat(created)
            age_days = (now - created_dt).total_seconds() / 86400.0
        scores.append(min(1.0, age_days / self.config.min_account_age_days))

        # Follower/following ratio --------------------------------------
        followers = float(user.get("followers", 0) or 0)
        following = float(user.get("following", 0) or 0)
        ratio = followers / (following + 1.0)
        scores.append(min(1.0, ratio / self.config.min_follower_ratio))

        # Posting interval entropy --------------------------------------
        posts = list(user.get("posts", []))
        if len(posts) > 1:
            times = sorted(
                datetime.fromisoformat(p["timestamp"]).timestamp()
                for p in posts
                if "timestamp" in p
            )
            intervals = np.diff(times)
            if intervals.size:
                hist, _ = np.histogram(intervals, bins=min(10, intervals.size))
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob + 1e-9))
                max_entropy = math.log2(len(hist)) if len(hist) > 1 else 1.0
                scores.append(entropy / max_entropy)
            else:
                scores.append(0.0)
        else:
            scores.append(1.0)

        # Repeated text similarity --------------------------------------
        texts = [p.get("text", "") for p in posts if p.get("text")]
        if texts:
            unique_ratio = len(set(texts)) / len(texts)
            scores.append(unique_ratio)
        else:
            scores.append(1.0)

        # Language anomalies --------------------------------------------
        if texts:
            all_text = "".join(texts)
            non_ascii = sum(ord(c) > 127 for c in all_text)
            ratio = non_ascii / max(1, len(all_text))
            scores.append(1.0 - min(1.0, ratio / self.config.max_non_ascii_ratio))
        else:
            scores.append(1.0)

        # Blacklist terms -----------------------------------------------
        if texts and any(term in t.lower() for t in texts for term in self.config.blacklist):
            scores.append(0.0)
        else:
            scores.append(1.0)

        return float(max(0.0, min(1.0, float(np.mean(scores)))))

    # ------------------------------------------------------------------
    def edge_weight(self, event: Mapping[str, object]) -> float:
        """Return an edge weight multiplier for ``event``.

        If no user information is present the multiplier defaults to ``1.0``.
        """

        user = None
        if "user" in event and isinstance(event["user"], Mapping):
            user = event["user"]
        elif "user_features" in event and isinstance(event["user_features"], Mapping):
            user = event["user_features"]

        if user is None:
            return 1.0

        return self.score_account(user)
