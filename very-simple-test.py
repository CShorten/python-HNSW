from hnsw import HNSW
import numpy as np

def test_distance():
    hnsw = HNSW()
    a = np.array([1, 1])
    b = np.array([4, 5])
    assert hnsw._distance(a, b) == 5.0, "Test Failed: Incorrect distance calculation."

def test_search_layer():
    hnsw = HNSW()
    q = (1,1)
    hnsw.graph = {
        0: {
            (1, 1): [(2, 2), (3, 3)],
            (2, 2): [(1, 1), (3, 3)],
            (3, 3): [(1, 1), (2, 2)],
        }
    }
    neighbors = hnsw._search_layer(q, (1, 1), 2, 0)
    print(neighbors)
    
def test_search():
    hnsw = HNSW()
    q = (1,1)
    hnsw.graph = {
        0: {
            (1, 1): [(2, 2), (3, 3)],
            (2, 2): [(1, 1), (3, 3)],
            (3, 3): [(1, 1), (2, 2)],
        }
    }
    hnsw.get_enter_point = lambda: (1, 1)
    hnsw.get_layer = lambda _: 0
    result = hnsw.search(q, 2, 2)
    print(result)
    
if __name__ == "__main__":
    test_distance()
    test_search_layer()
    test_search()
    print("All tests passed.")
