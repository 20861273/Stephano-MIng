import numpy as np
from collections import namedtuple
from operator import itemgetter

def unvisited_neighbors(id):
    (x, y) = id.x, id.y
    # (right, up, left, down)
    results = [Point(x+1, y), Point(x-1, y), Point(x, y-1), Point(x, y+1), Point(x, y)]
    # This is done to prioritise straight paths
    # if (x + y) % 2 == 0: results.reverse()
    results = list(filter(in_bounds, results))
    results = list(filter(unvisited, results))
    results = list(filter(uncounted, results))
    # results = filter(self.passable, results)
    return results

def in_bounds(id):
    (x, y) = id
    return 0 <= x < 5 and 0 <= y < 5

def unvisited(id):
    (x, y) = id
    return grid[y,x] == 1

def uncounted(id):
    return id not in counted_children

def largest_possible_block(current, visited, n):
    # Initialize the stack with the current position and the remaining time steps
    stack = [(current, visited, n)]
    # Initialize the maximum number of visited blocks to zero
    max_blocks = 0
    # Start the DFS algorithm
    while stack:
        pos, visited, steps = stack.pop()
        # Check if the agent has run out of time or has visited all the blocks
        if steps < 0 or len(visited) == H * W:
            max_blocks = max(max_blocks, len(visited))
            continue
        # Check if the current position is already visited
        if pos in visited:
            continue
        # Mark the current position as visited
        visited.add(pos)
        # Explore the neighboring positions
        row, col = pos
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            r, c = row + dr, col + dc
            if 0 <= r < H and 0 <= c < W:
                stack.append(((r, c), visited.copy(), steps - 1))
    return max_blocks

def get_neighbors(node, WIDTH, HEIGHT):
    WIDTH, HEIGHT = node
    neighbors = []
    if WIDTH > 0:
        neighbors.append((WIDTH - 1, HEIGHT))
    if HEIGHT > 0:
        neighbors.append((WIDTH, HEIGHT - 1))
    if WIDTH < WIDTH - 1:
        neighbors.append((WIDTH + 1, HEIGHT))
    if HEIGHT < HEIGHT - 1:
        neighbors.append((WIDTH, HEIGHT + 1))
    return neighbors

def get_distance(end, start):
    return abs(start.x - end.x) + abs(start.y - end.y)



visited = [(0,0), (0,1), (1,0), (1,1)]

H, W = 5, 5
n = 6

Point = namedtuple('Point', 'x, y')

start = (0, 0)

# print(count_zero_groups(grid))

current = start

children = [0]
counted_children = []

potential_reward = largest_possible_block(current, visited, n)

print(potential_reward)



