import numpy as np
from collections import namedtuple
from operator import itemgetter

def get_reachable_blocks(height, width, start_pos, n):
    grid = np.zeros((height, width))
    max_distance = height + width - 2
    distance_factor = 2
    threshold_distance = int(max_distance * distance_factor)
    
    for i in range(height):
        for j in range(width):
            manhattan_dist = abs(i - start_pos[0]) + abs(j - start_pos[1])
            if manhattan_dist <= threshold_distance:
                prob_reach = 1 / (5 ** manhattan_dist)
                if prob_reach >= 0.01:
                    print(prob_reach)
                    num_reached = int(prob_reach * 100) # scale up probability to avoid rounding to zero
                    grid[i][j] = num_reached
            else:
                grid[i][j] = -1  # unreachable
            
    return grid

# Define environment parameters
Height = 10
Width = 10
start = (0, 0)
n = 6

# Calculate reachable blocks
reachable_blocks = get_reachable_blocks(Height, Width, start, n)

print(reachable_blocks)



