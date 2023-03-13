import numpy as np
from collections import namedtuple
from operator import itemgetter

def get_distance(end, start):
    return abs(start.x - end.x) + abs(start.y - end.y)

def find_groups(visited):
    groups = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            # if point has been visited then we do not calculate the distance
            if Point(x,y) in visited:
                continue
            
            
            # adds point to a group of unvisited blocks
            if len(groups) == 0:
                groups.append([Point(x,y)])
                continue
            min_group_distance = float('inf')
            for g_i, group in enumerate(groups):
                for p in group:
                    group_distance = get_distance(Point(x,y), p)
                    # if next to group add to group
                    if group_distance == 1:
                        groups[g_i].append(Point(x,y))
                        min_group_distance = 1
                        break
                    if group_distance < min_group_distance:
                        min_group_distance = group_distance
            # add to new group
            if min_group_distance != 1:
                groups.append([Point(x,y)])

    return groups

def potential_reward_func(groups, current, start, t):
    distances = {}
    distances_to_start = {}
    distances_from_c = {}
    min_distance = float('inf')
    max_PR = 0

    # get closest unvisited block (keeping return to start in mind) and adds all distances to dictionaries for later use
    for g_i, group in enumerate(groups):
        group_max_PR = 0
        for p_i, p in enumerate(group):
            group_explored = False
            distances_to_start[p] = get_distance(p, start)
            distances_from_c[p] = get_distance(p, current)
            distances[p] = distances_to_start[p] + distances_from_c[p]

            # potential reward calculation
            remainder_steps = termination_time - distances[p] - t

            # can't reach
            if remainder_steps < 0:
                continue

            # if the size of the group is smaller than the remainder steps the potential reward can only be as large as the gorup
            if len(group) > remainder_steps:
                group_max_PR = remainder_steps
            else:
                group_max_PR = len(group)
            if distances_from_c[p] == 0:
                group_max_PR += 1

            if distances[p] < min_distance and group_max_PR > max_PR:

                min_distance = distances[p]
                closest_group = g_i
                p_index = p_i
                max_PR = group_max_PR

                
    return closest_group, p_index, max_PR

def find_largest_estimated_reward(visited, next_node, t):
    track_visited = [[False] * HEIGHT for _ in range(WIDTH)]
    groups = []
    distance = {}
    dist_to_uv = {}
    dist_to_s = {}
    # get group with minumum distance to next_node
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if Point(x,y) not in visited and not track_visited[x][y]:
                group_size = 0
                min_distance = float('inf')
                min_dist_to = float('inf')
                min_dist_pos = None
                stack = [(x, y)]
                while stack:
                    x, y = stack.pop()
                    if not track_visited[x][y]:
                        track_visited[x][y] = True
                        group_size += 1
                        dist_to_uv[Point(x,y)] = get_distance(Point(x,y), next_node) # plus 1 is for distance from current to next_node
                        dist_to_s[Point(x,y)] = get_distance(Point(x,y), start)
                        distance[Point(x,y)] = dist_to_uv[Point(x,y)] + dist_to_s[Point(x,y)]
                        if distance[Point(x,y)] <= min_distance and dist_to_uv[Point(x,y)] < min_dist_to:
                            min_dist_to = dist_to_uv[Point(x,y)]
                            min_distance = distance[Point(x,y)]
                            min_dist_pos = Point(x,y)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and Point(nx,ny) not in visited and not track_visited[nx][ny]:
                            stack.append((nx, ny))
                groups.append((group_size, min_dist_pos, min_distance))
    
    groups = sorted(groups, reverse=True)
        
    if not bool(groups):
        return None
    
    PR = 0
    for i, group in enumerate(groups):
        # potential reward calculation
        remainder_steps = termination_time - group[2] - t
        # can't reach
        if remainder_steps < 0:
            continue
        # if the size of the group is smaller than the remainder steps the potential reward can only be as large as the gorup
        if group[0] > remainder_steps:
            group_max_PR = remainder_steps
        else:
            group_max_PR = group[0]
        # if next node is closest unvisited block in group
        if not dist_to_uv[group[1]] == 0:
            group_max_PR -= 1

        if group_max_PR > PR:
            PR = group_max_PR
    
    if PR < 0:
        return None
    return PR, remainder_steps, group

# Define environment parameters
Point = namedtuple('Point', 'x, y')
HEIGHT = 5
WIDTH = 5
start = Point(0, 0)
next = Point(1, 0)
termination_time = 6
t = 0
visited = [Point(0,0)]
# next = Point(2, 0)
# termination_time = 6
# t = 1
# visited = [Point(0,0), Point(1, 0)]

grid = np.zeros((HEIGHT, WIDTH))
for y in range(HEIGHT):
    for x in range(WIDTH):
        if Point(x,y) in visited:
            grid[y,x] = 1
print(grid)

distance, rs, g = find_largest_estimated_reward(visited, next, t)

print(distance, rs, g)

# # Group unvisited blocks
# groups = find_groups(visited)

# # find group that would give the maximum potential reward
# group_index, p_index, potential_reward = potential_reward_func(groups, next, start, t)

# print(grid, "\n", groups)

# print(group_index, groups[group_index][p_index], potential_reward)


# # find minimum distance within group and
#                     # checks if the maximum potential reward is smaller than how many unvisited blocks are in the group
#                     # because the maximum potential reward can only be as big as the group is
#                     if group_distance < min_group_distance and max_PR < len(group):
#                         min_group_distance = group_distance
#                         closest_point = p
#                         closest_group = g_i

#                         # potential reward calculation
#                         remainder_steps = termination_time - min_group_distance
#                         # if the size of the group is smaller than the remainder steps the potential reward can only be as large as the gorup
#                         if len(group) > remainder_steps:
#                             max_PR = remainder_steps
#                         else:
#                             max_PR = len(group)

#                         groups[closest_group].append(Point(x,y))