import numpy as np
from hnsw import HNSW
import matplotlib.pyplot as plt

def generate_random_points(num_points):
    x = np.random.uniform(low=0, high=20, size=num_points)
    y = np.random.uniform(low=0, high=20, size=num_points)
    ids = [x for x in range(num_points)]
    labels = [id % 6 for id in ids]
    points = np.column_stack((x, y))
    
    return points, ids, labels

def plot_points(x, y, ids, point_colors, graph):
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=point_colors)

    for i, txt in enumerate(ids):
        ax.annotate(txt, (x[i], y[i]))


    for id in ids:
        id_x = x[id]
        id_y = y[id]
        neighbors = graph[0][id]
        for neighbor in neighbors:
            neighbor_x = x[neighbor]
            neighbor_y = y[neighbor]
            plt.plot([id_x, neighbor_x], [id_y, neighbor_y], linestyle='-', marker='o')
    

    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    plt.title('Random Points in a 20x20 2D Space with Layer 0 HNSW Graph Structure')
    plt.show()

def test_hnsw():
    num_points = 20
    M = 4
    M_max = 4
    efConstruction = 4
    ef = 16

    points, ids, labels = generate_random_points(num_points)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    point_colors = [colors[label] for label in labels]

    hnsw = HNSW()

    for i, point in enumerate(points):
        print(f"Inserting node... {ids[i]}")
        hnsw._insert_node(ids[i], point, M, M_max, efConstruction)
        hnsw._pretty_print_graph()
        
    query = np.array([0, 0])
    nearest_neighbors = hnsw.search(query, K=3, ef=ef)
    for neighbor in nearest_neighbors:
        print(f"Neighbor: {neighbor}, coords: {points[neighbor]}")

    plot_points(points[:, 0], points[:, 1], ids, point_colors, hnsw.graph)

if __name__ == "__main__":
    test_hnsw()
