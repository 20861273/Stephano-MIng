import numpy as np
from collections import namedtuple
from operator import itemgetter

def find_groups(grid, start):
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    groups = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                group_size = 0
                min_distance = float('inf')
                group_positions = []
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if not visited[x][y]:
                        visited[x][y] = True
                        group_size += 1
                        group_positions.append((x, y))
                        distance = abs(x - start[0]) + abs(y - start[1])
                        if distance < min_distance:
                            min_distance = distance
                            min_dist_pos = (x,y)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and not visited[nx][ny]:
                            stack.append((nx, ny))
                groups.append((group_size, min_dist_pos, group_positions))
    return sorted(groups, reverse=True)

# Define environment parameters
Height = 10
Width = 10
start = (4, 0)
n = 6
grid = [[0,0,0,0,0],
        [0,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,0,0],
        [1,1,0,1,1]]

# Calculate reachable blocks
reachable_blocks = find_groups(grid, start)

print(reachable_blocks)

for i, distance in enumerate(reachable_blocks):
    print(distance[1])


