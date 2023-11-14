from enclosed_space_checker import Enclosed_space_check
import numpy as np
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
import time
from collections import namedtuple
Point = namedtuple('Point', 'x, y')

grid = np.zeros((5,5))

i = np.argwhere(grid == 0)
np.random.shuffle(i)
s = int(len(i)-20)
for j in range(s):
    grid[i[j,1], i[j,0]] = 1

i = np.argwhere(grid == 0)
np.random.shuffle(i)
s = int(len(i)-15)
for j in range(s):
    grid[i[j,1], i[j,0]] = 2

print(grid)

pos = Point(1,1)

right_is_boundary = pos.x == 5 - 1
left_is_boundary = pos.x == 0
top_is_boundary = pos.y == 0
bottom_is_boundary = pos.y == 5 - 1

surroundings = []

surroundings.append(right_is_boundary or grid[pos.y][pos.x+1] == 1 or grid[pos.y][pos.x+1] == 2)
surroundings.append(left_is_boundary or grid[pos.y][pos.x-1] == 1 or grid[pos.y][pos.x-1] == 2)
surroundings.append(top_is_boundary or grid[pos.y-1][pos.x] == 1 or grid[pos.y-1][pos.x] == 2)
surroundings.append(bottom_is_boundary or grid[pos.y+1][pos.x] == 1 or grid[pos.y+1][pos.x] == 2)

print(surroundings)
breakpoint