import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

# Generate random points
np.random.seed(42)
points = np.random.rand(20, 2)

# Perform Delaunay triangulation
tri = Delaunay(points)

# Create Voronoi diagram from the Delaunay triangulation
vor = Voronoi(points)

# Plot the Delaunay triangulation and Voronoi diagram
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot Delaunay triangulation
ax[0].triplot(points[:, 0], points[:, 1], tri.simplices)
ax[0].scatter(points[:, 0], points[:, 1], c='r', marker='.')
ax[0].set_title("Delaunay Triangulation")

# Plot Voronoi diagram
voronoi_plot_2d(vor, ax=ax[1], point_size=10)
ax[1].set_title("Voronoi Diagram")

plt.show()
