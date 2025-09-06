import torch

from datetime import datetime
from typing import Dict, List, Optional
from data_pipeline.transformers.event_parser import parse_event
from data_pipeline.storage.decay import apply_time_decay
from utils.validation import validate_graph
from torch_geometric.data import TemporalData
from model.inference.spam_filter import SpamScorer


class GraphBuilder:
    """Builds a temporal graph from event data usable by PyG TGN model.

    Events are streamed in and converted into lists of source nodes, destination nodes, timestamps and edge attributes. Nodes are tracked
    via internal mapping from their string identifiers to an integer index so that the graph can grow beyond memory limits without starting a full adjacency structure.
    """

    def __init__(
        self,
        reference_time: Optional[datetime] = None,
        *,
        spam_scorer: SpamScorer | None = None,
    ) -> None:
        self.reference_time = reference_time or datetime.now()
        self.spam_scorer = spam_scorer

        # Edge storage for TemporalData
        self.src: List[int] = []
        self.dst: List[int] = []
        self.t: List[float] = []
        self.msg: List[List[float]] = []

        # Node bookkeeping
        self.node_map: Dict[str, int] = {}
        self.node_types: Dict[int, str] = {}

    def process_event(self, event):
        """Parse an incoming event and append it to the temporal edge stream."""
        timestamp = datetime.fromisoformat(event["timestamp"])
        edges = parse_event(event)

        spam_multiplier = 1.0
        if self.spam_scorer is not None:
            spam_multiplier = self.spam_scorer.edge_weight(event)

        for _, source, target, label in edges:
            s_idx = self._add_node(source)
            t_idx = self._add_node(target)

            if s_idx is not None and t_idx is not None:
                weight = (
                    apply_time_decay(timestamp, self.reference_time) * spam_multiplier
                )
                self.src.append(s_idx)
                self.dst.append(t_idx)
                self.t.append(timestamp.timestamp())
                self.msg.append([weight])
                print(f"{source} -> {target} ({label}) - weight={weight:.4f}")

    def _add_node(self, node: Optional[str]) -> Optional[int]:
        """Register a node string and return its integer index."""
        if node is None:
            return None
        if node not in self.node_map:
            idx = len(self.node_map)
            self.node_map[node] = idx
            self.node_types[idx] = self._infer_type(node)
        return self.node_map[node]

    def _infer_type(self, node: str) -> str:
        if node.startswith("h_"):
            return "hashtag"
        if node.startswith("yt_v"):
            return "youtube_video"
        # if node.startswith("tk_v"):
        #     return "tiktok_video"
        if node.startswith("trend_"):
            return "trend"
        if node.startswith("ctx_"):
            return "context"
        # TODO (for production): Add more types
        return "unknown"

    def validate(self):
        validate_graph(self.src, self.dst)

    def to_temporal_data(self) -> TemporalData:
        """Return collected edges as a :class: `TemporalData` object."""
        if not self.src:
            return TemporalData
        return TemporalData(
            src=torch.tensor(self.src, dtype=torch.long),
            dst=torch.tensor(self.dst, dtype=torch.long),
            t=torch.tensor(self.t, dtype=torch.float),
            msg=torch.tensor(self.msg, dtype=torch.float),
        )
