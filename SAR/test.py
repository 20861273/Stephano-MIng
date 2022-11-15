from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math
import time

Point = namedtuple('Point', 'y, x')

grid = np.zeros((4,3))
starting_pos = [Point(0,0)]*2
state = [0]*2

grid.fill(0)
grid_cnt = np.zeros((144,))

# Setup agent
# Set robot(s) start position
for i in range(2000000):
    indices = np.argwhere(grid == 0)
    np.random.shuffle(indices)
    starting_pos[0] = Point(indices[0,0], indices[0,1])
    starting_pos[1] = Point(indices[1,0], indices[1,1])
    state[0] = starting_pos[0].y*grid.shape[1] + starting_pos[0].x
    state[1] = starting_pos[1].y*grid.shape[1] + starting_pos[1].x

    s = [(state[0]*grid.shape[0]*grid.shape[1] + state[1]),
        (state[1]*grid.shape[0]*grid.shape[1] + state[0])]

    grid_cnt[s[0]] += 1
    grid_cnt[s[1]] += 1

print(grid_cnt)