def validate_graph(G):
    """ Validate the integrity of the graph structure.
    Checks for self-loops, edge types, and node assignments.
    """
    try:
        assert not any(u == v for u, v in G.edges()), "ERROR: Graph has self-loops!"
        assert all(0 <= d['weight'] <= 1 for _, _, d in G.edges(data=True)), "ERROR: Edge weights out of range!"
        assert all('type' in d for _, d in G.nodes(data=True)), "ERROR: Some nodes lack a type!"
        print("SUCCESS: Graph passed all integrity checks.")
    except AssertionError as e:
        print(str(e))