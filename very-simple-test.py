import numpy as np
from hnsw import HNSW

def test_hnsw():
    # Create an HNSW instance with default parameters
    hnsw = HNSW()
    M = 16
    M_max = 32
    efConstruction = 16
    mL = 2
    ef = 16

    # Add some nodes (vectors) to the graph
    hnsw._insert_node(0, np.array([0.5, 0.5]), M, M_max, efConstruction, mL)
    hnsw._pretty_print_graph()
    hnsw._insert_node(1, np.array([1.0, 1.0]), M, M_max, efConstruction, mL)
    hnsw._pretty_print_graph()
    hnsw._insert_node(2, np.array([2.0, 2.0]), M, M_max, efConstruction, mL)
    hnsw._pretty_print_graph()
    hnsw._insert_node(3, np.array([3.0, 3.0]), M, M_max, efConstruction, mL)
    hnsw._pretty_print_graph()
    hnsw._insert_node(4, np.array([4.0, 4.0]), M, M_max, efConstruction, mL)

    # Perform a search for the nearest neighbors of a query vector
    query = np.array([3.8, 3.8])
    nearest_neighbors = hnsw.search(query, K=3, ef=ef)

    # Check if the search result is correct
    print(nearest_neighbors)

if __name__ == "__main__":
    test_hnsw()
