from hnsw import HNSW
import unittest

class TestHNSW():
    def __init__(self):
        self.graph = HNSW()
        self.graph.layers = [
            {  # Layer 0
                (1,1): [(2, 2), (3, 3)],
                (2,2): [(1, 1), (3, 3), (4, 4)],
                (3,3): [(1, 1), (2, 2), (4, 4)],
                (4,4): [(2, 2), (3, 3)]
            },
            {  # Layer 1
                (1,1): [(3, 3)],
                (3,3): [(1, 1), (4, 4)],
                (4,4): [(3, 3)]
            }
        ]

    def test_search_layer(self):
        query = (1, 1)
        entry_point = (3,3)
        ef = 2
        layer_num = 0
        expected_nearest_neighbors = [(1, 1), (2, 2)]

        result = self.graph._search_layer(query, entry_point, ef, layer_num)
        print(f"Result - {result}")
        print(f"Expected - {expected_nearest_neighbors}")
        
test = TestHNSW()
test.test_search_layer()