import numpy as np

# load points and labels from disk
data = np.load('points.npz')
x = data['x']
y = data['y']
ids = data['ids']
labels = data['labels']

# print the first five points and their labels
for i in range(5):
    print(f"Point {ids[i]}: ({x[i]:.2f}, {y[i]:.2f}), Label: {labels[i]}")
