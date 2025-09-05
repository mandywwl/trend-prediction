"""Test helper utilities."""

def create_sample_graph_data():
    """Create sample graph data for testing."""
    import numpy as np
    
    # Simple test graph with 5 nodes and 10 edges
    src = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    dst = np.array([1, 2, 3, 4, 0, 2, 3, 4, 0, 1])
    t = np.arange(10, dtype=np.float64)
    edge_attr = np.random.randn(10, 32).astype(np.float32)
    node_features = np.random.randn(5, 16).astype(np.float32)
    
    return {
        "src": src,
        "dst": dst, 
        "t": t,
        "edge_attr": edge_attr,
        "node_features": node_features
    }