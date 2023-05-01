import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import recall_score
from hnsw import HNSW
import random

def generate_bimodal_gaussian_data(n_points, dim, mu1, sigma1, mu2, sigma2):
    data = []
    for _ in range(n_points):
        choice = random.choice([0, 1])
        if choice == 0:
            point = np.random.normal(mu1, sigma1, dim)
        else:
            point = np.random.normal(mu2, sigma2, dim)
        data.append(point)
    return data

def calculate_recall(found_neighbors, true_neighbors):
    count = 0
    for found, true in zip(found_neighbors, true_neighbors):
        if len(set(found).intersection(set(true))) > 0:
            count += 1
    return count / len(found_neighbors)

def test_hnsw(max_connections, max_layers, ef, efConstruction):
    n_points = 1000
    dim = 2
    mu1, sigma1 = [0, 0], [1, 1]
    mu2, sigma2 = [10, 10], [1, 1]
    K = 10

    data = generate_bimodal_gaussian_data(n_points, dim, mu1, sigma1, mu2, sigma2)

    hnsw = HNSW(max_connections=max_connections, max_layers=max_layers, ef=ef, efConstruction=efConstruction)
        
    for i, point in enumerate(data):
        hnsw._insert_node(i, point)

    found_neighbors = []
    true_neighbors = []

    for i, query in enumerate(data):
        true_neighbor_indices = np.argsort([np.linalg.norm(query - point) for point in data])[1:K+1]
        true_neighbors.append(true_neighbor_indices)
        found_neighbor_indices = hnsw.search(query, K, 100)
        found_neighbors.append(found_neighbor_indices)
    

    recall = calculate_recall(found_neighbors, true_neighbors)
    print(f"Tested, max_connections={max_connections}")
    print(f"Tested, max_layers={max_layers}")
    print(f"Tested, ef={ef}")
    print(f"Tested, efConstruction={efConstruction}")
    print(f"Resulting Recall: {recall * 100:.2f}%")

# Example usage:
max_connections_list = [16, 32, 64, 128, 256, 512]
max_layers_list = [3,4,5]
ef_list = [16,32,64,128,256,512]
efConstruction_list = [16,32,64,128,256,512]

import time

for max_connections in max_connections_list:
    for max_layers in max_layers_list:
        for ef in ef_list:
            for efConstruction in efConstruction_list:
                start = time.time()
                test_hnsw(max_connections=max_connections, 
                          max_layers=max_layers, 
                          ef=ef,
                          efConstruction=efConstruction)
                print(f"Tested in {time.time() - start} seconds.")
