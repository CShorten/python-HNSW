import h5py
from hnsw import HNSW
import numpy as np

# Open the HDF5 file for reading
with h5py.File('sift-128-euclidean.hdf5', 'r') as file:

    # Print the keys of the top-level groups
    print(list(file.keys()))

    # Access the 'distances' dataset in the top-level group
    train_data = np.array(file['train'])
    test_data = np.array(file['test'])
    distances = file['distances']
    neighbors = file['neighbors']

# Initialize HNSW graph with desired parameters
hnsw = HNSW(max_connections=16, max_layers=2, ef=16, efConstruction=16)

# Insert the training data nodes into the HNSW graph
num_train_vectors = train_data.shape[0]
for i in range(num_train_vectors):
    if i % 1_000 == 999:
        print(f"Inserted {i} vectors into the graph.")
    vector = train_data[i]
    hnsw._insert_node(i, vector)

# Example query on the HNSW graph
query_vector = test_data[0]
K = 10  # number of nearest neighbors to return
ef = 50  # size of the dynamic candidate list
result = hnsw.search(query_vector, K, ef)

print(f"Nearest neighbors for query vector {query_vector}: {result}")
