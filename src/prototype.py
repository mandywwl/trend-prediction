import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from utils import parse_event, apply_time_decay, validate_graph


#---- Load mock events ----
with open("data/mock_tweets.json", "r") as f:
    events = json.load(f)

#---- Build directed graph ----
G = nx.DiGraph()
reference_time = datetime.fromisoformat(events[0]["timestamp"])

for i, event in enumerate(sorted(events, key=lambda x: x["timestamp"])):
    timestamp = datetime.fromisoformat(event["timestamp"])
    _, source, target, label = parse_event(event)

    for node in [source, target]:
        if node and not G.has_node(node):
            G.add_node(node, type="user" if node.startswith("u") else "tweet" if node.startswith("t") else "hashtag")

    G.add_edge(source, target, label=label, timestamp=timestamp)
    G[source][target]["weight"] = apply_time_decay(timestamp, reference_time)

    #---- Simulate real-time streaming ----
    print(f"[{i+1}/{len(events)}] {source} -> {target} ({label}) - weight={G[source][target]['weight']:.4f}")
    time.sleep(1)

#---- Visualise graph ----
def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    node_types = nx.get_node_attributes(G, "type")
    colors = {"user": "lightgreen", "tweet": "skyblue", "hashtag": "orange"}
    node_colours = [colors.get(node_types[n], "grey") for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15)

    edge_labels = {
        (u, v): f"{d['label']} ({d['weight']:.2f})"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Graph with Time-Decayed Edge Weights")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

draw_graph(G)


#---- Log edge weights over time ----
timestamps = []
weights = []
for _, _, d in G.edges(data=True):
    timestamps.append(d["timestamp"])
    weights.append(d["weight"])

plt.figure(figsize=(8, 4))
plt.plot(timestamps, weights, marker="o")
plt.xticks(rotation=45)
plt.ylabel("Decayed Weight")
plt.xlabel("Timestamp")
plt.title("Edge Influence Decay Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

#---- Validate graph structure ----
validate_graph(G)

#---- Save graph to file ----
with open("export/demo_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
