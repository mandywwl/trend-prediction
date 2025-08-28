def validate_graph(src, dst):
    """ Simple validation for the temporal edge stream
    Currently checks only for self-loops to keep memory usage minimal.
    """
    try:
        assert all(s != d for s, d in zip(src, dst)), "ERROR: Graph has self-loops!"
        print("SUCCESS: Graph passed all integrity checks.")
    except AssertionError as e:
        print(str(e))