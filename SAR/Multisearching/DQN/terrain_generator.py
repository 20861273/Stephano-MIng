import numpy as np
from dqn_environment import Environment, HEIGHT, WIDTH
import matplotlib
import matplotlib.pyplot as plot
import math
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('\perlin_noise\perlin_noise')
from perlin_noise import PerlinNoise

HEIGHT = 50
WIDTH = 50

largest = 0
smallest = 0

noise1 = PerlinNoise(5)
noise2 = PerlinNoise(10)
noise3 = PerlinNoise(15)
value = np.zeros((HEIGHT, WIDTH))
# generates distribution
for y in range(HEIGHT):
    for x in range(WIDTH):      
        nx = x/WIDTH# - 0.5
        ny = y/HEIGHT# - 0.5
        e = noise1([nx, ny])
        e += 0.5 * noise2([nx, ny])
        e += 0.25 * noise3([nx, ny])

        e = e / (1 + 0.5 + 0.25)

        if e > largest: largest = e
        if e < smallest: smallest = e

        value[y][x] = e

# scales distribution [0,1]
for y in range(HEIGHT):
    for x in range(WIDTH):
        value[y][x] = (value[y][x] - smallest) / (largest - smallest)

# redistributes to make flat valleys
for y in range(HEIGHT):
    for x in range(WIDTH):
        value[y][x] = (value[y][x] - smallest) / (largest - smallest)
        value[y][x] = math.pow(value[y][x], 1)
        if value[y][x] > largest: largest = value[y][x]
        if value[y][x] < smallest: smallest = value[y][x]

# scales distribution [0,1]
for y in range(HEIGHT):
    for x in range(WIDTH):
        value[y][x] = (value[y][x] - smallest) / (largest - smallest)

print(value)

# plot.imshow(value, cmap='gray')
# plot.colorbar()

# plot.show()

lin_x = np.linspace(0,1,value.shape[0],endpoint=False)
lin_y = np.linspace(0,1,value.shape[1],endpoint=False)
x,y = np.meshgrid(lin_x,lin_y)

# fig = matplotlib.pyplot.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(x,y,value,cmap='gray')
# ax.set_zlim(0,1)

# plot.show()


# Calculate the mean
mean = np.mean(value)

print("mean",mean)

# Split data into two groups: above and below mean
above_mean = value[value > mean]
below_mean = value[value < mean]

min_below = np.min(below_mean)
min_above = np.min(above_mean)
max_below = np.max(below_mean)
max_above = np.max(above_mean)

max_diff_above = max_above - mean
max_diff_below = mean - min_above
max_diff = max(max_diff_above, max_diff_below)

# difference calculations
prob_dist = np.zeros(value.shape)
for y in range(HEIGHT):
    for x in range(WIDTH):
        if value[y][x] > mean: difference = max_above - value[y][x]
        if value[y][x] < mean: difference = mean - value[y][x]
        prob_dist[y][x] = (difference - 0) / (max_diff - 0)
        if value[y][x] == mean: prob_dist[y][x] = 1.0

print(prob_dist)

fig, ax = plot.subplots(1, 3, figsize=(15, 5))

im1 = ax[0].imshow(value, cmap='gray')
fig.colorbar(im1, ax=ax[0])
ax[0].set_title('2D terrain')

lin_x = np.linspace(0,1,value.shape[0],endpoint=False)
lin_y = np.linspace(0,1,value.shape[1],endpoint=False)
x,y = np.meshgrid(lin_x,lin_y)

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(x, y, value, cmap='gray')
ax2.set_zlim(0, 1)
ax2.set_title('3D terrain')

im2 = ax[2].imshow(prob_dist)
fig.colorbar(im2, ax=ax[2])
ax[2].set_title('Probability distribution\nMean: %.2f' %(mean))

# plot.show()

plot.savefig("fig.png")