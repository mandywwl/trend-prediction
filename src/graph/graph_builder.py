import networkx as nx
from datetime import datetime
from utils.event_parsing import parse_event
from graph.time_decay import apply_time_decay
from utils.validation import validate_graph


class GraphBuilder:
    def __init__(self, reference_time=None):
        self.G = nx.DiGraph()
        self.reference_time = reference_time or datetime.now()

    def process_event(self, event):
        timestamp = datetime.fromisoformat(event["timestamp"])
        edges = parse_event(event) 
        for _, source, target, label in edges:
            for node in [source, target]:
                if node and not self.G.has_node(node):
                    self.G.add_node(node, type=self._infer_type(node), platform=event.get("source"))
            if source and target:
                self.G.add_edge(source, target, label=label, timestamp=timestamp, platform=event.get("source"))
                self.G[source][target]["weight"] = apply_time_decay(timestamp, self.reference_time)
                print(f"{source} -> {target} ({label}) - weight={self.G[source][target]['weight']:.4f}")


    def _infer_type(self, node):
        if node is None:
            return "unknown"
        if node.startswith("u"):
            return "user"
        if node.startswith("t"):
            return "tweet"
        if node.startswith("h_"):
            return "hashtag"
        if node.startswith("yt_v"):
            return "youtube_video"
        if node.startswith("tk_v"):
            return "tiktok_video"
        if node.startswith("trend_"):
            return "trend"
        # TODO: Add more types
        return "unknown"

    def validate(self):
        validate_graph(self.G)
