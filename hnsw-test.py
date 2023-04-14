from hnsw import HNSW
import numpy as np

def test_insert_and_search():
    num_points = 30
    x = np.random.uniform(low=0, high=20, size=num_points)
    y = np.random.uniform(low=0, high=20, size=num_points)

    points = list(zip(x, y))

    # Initialize HNSW
    hnsw = HNSW()

    # Insert the points into the HNSW graph
    for point in points:
        hnsw._insert_node(point, hnsw.max_connections, hnsw.max_connections, hnsw.efConstruction, hnsw.max_layers)

    # Generate a random query point
    query_point = np.random.uniform(low=0, high=20, size=2)

    # Find the nearest neighbor using HNSW
    hnsw_neighbors = hnsw.search(query_point, 1, hnsw.ef)

    # Find the nearest neighbor using brute force method
    brute_force_neighbors = sorted(points, key=lambda x: hnsw._distance(query_point, x))[:1]

    assert hnsw_neighbors == brute_force_neighbors, "Test Failed: HNSW nearest neighbor doesn't match brute force result."

test_insert_and_search()
