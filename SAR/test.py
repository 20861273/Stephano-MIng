import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.pyplot import cm
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from datetime import datetime

from collections import namedtuple

def check_surrounding_cells(grid, locations):
    num_rows = len(grid)
    num_cols = len(grid[0])
    num_ones = len(locations)

    surroundings = []
    for i in range(num_ones):
        y,x = locations[i]

        # Check if the surrounding cells are on the edge
        right_is_boundary = x == num_cols - 1
        left_is_boundary = x == 0
        top_is_boundary = y == 0
        bottom_is_boundary = y == num_rows - 1

        surroundings.append([
            right_is_boundary or (grid[y][x+1] == 2 if not right_is_boundary else True),
            left_is_boundary or (grid[y][x-1] == 2 if not left_is_boundary else True),
            top_is_boundary or (grid[y-1][x] == 2 if not top_is_boundary else True),
            bottom_is_boundary or (grid[y+1][x] == 2 if not bottom_is_boundary else True)
        ])

    return surroundings

grid = np.zeros((4,4))
locations = [(1,2),(2,3),(0,3)]
for i in locations:
    grid[i] = 1

grid[1,3] = 2

print(grid)

cells = check_surrounding_cells(grid, locations)

print(cells)