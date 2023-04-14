import matplotlib.pyplot as plt
import numpy as np

# generate 30 random points in a 20x20 2D space
num_points = 30
x = np.random.uniform(low=0, high=20, size=num_points)
y = np.random.uniform(low=0, high=20, size=num_points)

# assign random id to each point
ids = np.random.randint(low=1, high=100, size=num_points)

# set label of each point to be % 6
labels = ids % 6

# color the points based on their labels
colors = ['r', 'g', 'b', 'c', 'm', 'y']
point_colors = [colors[label] for label in labels]

# plot the points
fig, ax = plt.subplots()
ax.scatter(x, y, c=point_colors)

# add labels to each point
for i, txt in enumerate(ids):
    ax.annotate(txt, (x[i], y[i]))

# set x and y axis limits
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])

# save points and labels to disk
np.savez('points.npz', x=x, y=y, ids=ids, labels=labels)
# set title and show plot
plt.title('30 Random Points in a 20x20 2D Space')
plt.show()
