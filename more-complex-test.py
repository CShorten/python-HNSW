import numpy as np
from hnsw import HNSW
from filteredrobustprune import experimentalHNSW
import matplotlib.pyplot as plt
import networkx as nx
import random

def generate_random_points(num_points):
    x = np.random.uniform(low=0, high=20, size=num_points)
    y = np.random.uniform(low=0, high=20, size=num_points)
    ids = [x for x in range(num_points)]
    labels = [id % 6 for id in ids]
    points = np.column_stack((x, y))
    
    return points, ids, labels

def construct_graph(num_points, points, ids, labels):
    M = 32
    M_max = 32
    efConstruction = 32
    ef = 16

    #hnsw = HNSW()
    hnsw = experimentalHNSW()

    for i, point in enumerate(points):
        print(f"Inserting node... {ids[i]}")
        hnsw._insert_node(ids[i], point, M, M_max, efConstruction, labels[i])
        hnsw._pretty_print_graph()

    return hnsw.graph[0]

def plot_points(x, y, ids, point_colors, graph):
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=point_colors)

    for i, txt in enumerate(ids):
        ax.annotate(txt, (x[i], y[i]), fontsize=15)

    for id, neighbors in graph.items():
        id_x = x[id]
        id_y = y[id]
        for neighbor in neighbors:
            neighbor_x = x[neighbor]
            neighbor_y = y[neighbor]
            plt.plot([id_x, neighbor_x], [id_y, neighbor_y], linestyle='-', marker='o')
    
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    plt.title('Random Points in a 20x20 2D Space with Layer 0 HNSW Graph Structure')
    plt.show()

def plot_graph(graph):
    G = nx.Graph()
    for node, neighbors in graph.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='orange', edge_color='gray', node_size=1500, font_size=12)

    # Show the plot
    plt.show()

def analyze_graph(graph, allow_list):
    visited = set()
    '''
    {0: [5, 6, 12, 1], 1: [12, 11, 0, 5], 2: [9, 16, 3, 1], 3: [9, 16, 2, 14], 4: [8, 5, 6, 0], 5: [0, 6, 1, 18], 6: [0, 5, 12, 8], 7: [11, 1, 0, 13], 8: [6, 0, 18, 4], 9: [3, 2, 1, 0], 10: [6, 0, 1, 3], 11: [7, 1, 12, 0], 12: [0, 1, 6, 18], 13: [15, 7, 1, 11], 14: [3, 12, 5, 2], 15: [19, 12, 1, 13], 16: [3, 2, 1, 12], 17: [4, 12, 8, 0], 18: [5, 12, 8, 0], 19: [15, 18, 12, 5]}
    '''
    allowed = set(allow_list)

    random_ep = random.choice(allow_list)
    print(f"Random entry point from allow_list - {random_ep}")

    dfs_queue = [random_ep]
    visited.add(random_ep)
    
    while len(dfs_queue) > 0:
        node = dfs_queue[0]
        for neighbor in graph[node]:
            if neighbor not in visited and neighbor in allowed:
                dfs_queue.append(neighbor)
        visited.add(node)
        dfs_queue.remove(node)

    common_elements = [elem for elem in visited if elem in allow_list]
    return len(common_elements) / len(allow_list) * 100

def test_hnsw():
    num_points = 50
    points, ids, labels = generate_random_points(num_points)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    point_colors = [colors[label] for label in labels]

    layer_zero_graph = construct_graph(num_points, points, ids, labels)

    #print(layer_zero_graph)
    nodes_to_check = [x for x in range(50) if x % 6 == 1]
    print(analyze_graph(layer_zero_graph, nodes_to_check))

    #plot_graph(layer_zero_graph)
    plot_points(points[:, 0], points[:, 1], ids, point_colors, layer_zero_graph)

if __name__ == "__main__":
    test_hnsw()
