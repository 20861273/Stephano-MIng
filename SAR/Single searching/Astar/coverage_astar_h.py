from operator import itemgetter
import numpy as np

from astar_environment import Environment, HEIGHT, WIDTH, Point
from save_results import print_results

from datetime import datetime
import time
import os
import matplotlib.pyplot as plt
import json

import time

import shutil

termination_time = 50
HH = False
draw = True

PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'Astar')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path)
date_and_time = datetime.now()
save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
if not os.path.exists(save_path): os.makedirs(save_path)

def write_json(lst, path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "w") as f:
        json.dump(lst, f)

def read_json(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        return json.load(f)

class GridWithWeights(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.weights = {}
        self.visited_blocks = set()

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.weights

    def cost(self, current, next_node, closed_set, visited_list):
        branch = reconstruct_path(closed_set, current)
        # reachable, distance, heading_home = closest_unvisited(branch, visited_list, current, next_node, current[7])
        cost = 0
        explored = 0

        visited = []
        for row in visited_list:
            for p in row:
                visited.append(p)
        
        visited = visited + branch

        estimated_reward = find_largest_estimated_reward(visited, next_node, current)

        if estimated_reward == None:
            return None

        # if (not reachable and distance == None):
        #     return None, False, None, None, explored
        
        # if next_node in branch:
        #     cost -= 1
        # if len([True for row in visited_list if next_node in row]) != 0:
        #     cost -= 1
        if next_node not in branch and not len([True for row in visited_list if next_node in row]) != 0:
            cost += 1
            explored += 1

        return estimated_reward#, reachable, distance, heading_home, explored

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x-1, y), Point(x, y-1), Point(x, y+1), Point(x, y)]
        # This is done to prioritise straight paths
        # if (x + y) % 2 == 0: results.reverse()
        results = list(filter(self.in_bounds, results))
        # results = filter(self.passable, results)
        return results

def within_range(start, current):
    dist = abs(current[4].x - start.x) + abs(current[4].y - start.y)
    if dist <= termination_time-current[2]:
        return True
    return False

def find_largest_estimated_reward(visited, next_node, current):
    track_visited = [[False] * HEIGHT for _ in range(WIDTH)]
    groups = []
    distances = {}
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if Point(x,y) not in visited and not track_visited[x][y]:
                group_size = 0
                min_distance = float('inf')
                min_dist_pos = None
                stack = [(x, y)]
                while stack:
                    x, y = stack.pop()
                    if not track_visited[x][y]:
                        track_visited[x][y] = True
                        group_size += 1
                        dist_to_uv = get_distance(Point(x,y), next_node)
                        # dist_to_s = get_distance(Point(x,y), env.starting_pos)
                        distance = dist_to_uv #+ dist_to_s
                        if distance < min_distance:
                            min_distance = distance
                            min_dist_pos = (x,y)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and Point(nx,ny) not in visited and not track_visited[nx][ny]:
                            stack.append((nx, ny))
                groups.append((group_size, min_dist_pos, min_distance))
    
    groups = sorted(groups, reverse=True)

    if len(groups) == 1 and groups[0][1] == None:
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if Point(x,y) not in visited:
                    dist_to_uv = get_distance(Point(x,y), next_node)
                    dist_to_s = get_distance(Point(x,y), env.starting_pos)
                    distance = dist_to_uv + dist_to_s

                    if distance <= termination_time - current[2]:
                        distances[Point(x,y)] = distance
                        distances[Point(x,y)] = termination_time - current[2] - distance
    else:
        for i, group in enumerate(groups):
            distance = get_distance(Point(group[1][0], group[1][1]), env.starting_pos) + group[2]
            if distance <= termination_time - current[2]:
                distances[group[1]] = termination_time - current[2] - distance
        
    if not bool(distances):
        return 0
    return max(distances.values())

def closest_unvisited(branch, visited, current, next, explored):
    grid = env.grid.copy()
    grid.fill(0)
    distances = {}
    
    # fills visited blocks with 1's in grid
    for row in visited:
        for p in row:
            grid[p.y, p.x] = 1
    for p in branch:
        grid[p.y, p.x] = 1
    # gets the distance to all unvisited blocks
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y,x] == 0:
                dist_to_uv = get_distance(Point(x,y), next)
                dist_to_s = get_distance(Point(x,y), env.starting_pos)
                distance = dist_to_uv + dist_to_s

                if distance <= termination_time - current[2]:
                    distances[Point(x,y)] = distance
    
    
    # Normalize distance
    max_dist = (HEIGHT+WIDTH-2) * 2

    dist_next_to_s = get_distance(env.starting_pos, next)
    if dist_next_to_s >= termination_time - current[2]:
        dist_next_to_s = float('inf')
    
    # Check if unvisted block reachable
    if not bool(distances) or dist_next_to_s == float('inf'):
        # Can't get back to start
        if dist_next_to_s == float('inf') or explored == 0:
            return False, None, None
        # Can't get to unvisited block, but can return home
        else:
            return True, dist_next_to_s/max_dist/10, True
    else:
        # Unvisited block reachable
        min_distance =  distances[min(distances, key=distances.get)]/max_dist/10
        return True, min_distance, False

def get_distance(end, start):
    return abs(start.x - end.x) + abs(start.y - end.y)

def reconstruct_path(closed_set, current):
    current = current[1:5]
    path = [current[3]]
    while current[0] != -1:
        current = closed_set[current[0]]
        path.append(current[3])
    path.reverse()
    return path

def calc_cost(closed_set, ids):
    cost = 0
    for id in ids:
        costs = [closed_set[id]]
        current = closed_set[id]
        while True:
            if current[0] == -1: break
            current = closed_set[current[0]]
            costs.append(current)
        for c in costs: 
            if c[3] == closed_set[0][3]: cost += c[2]
    return cost

# A* algorithm
def a_star(graph, start, termination_time):
    times = [0]
    global HH
    debug = False
    visited = 0
    explored = 0
    visited_list = []
    closed_set = {}
    closed_set[0] = (-1, 0, 0, start)
    id = 1
    open_set = [(0, -1, 0, 0, start)]
    current = open_set[-1]
    last_ids = []
    while True:
        printer = print_results(env.grid, HEIGHT, WIDTH)
        # id, parent_id, time, reward, position, visited
        open_set = [(0, 0, 0, 0, start)]
        
        t = time.time() - stime
        while len(open_set) > 0:
        # while len(open_set) > 0 and t < 10:
            t = time.time() - stime
            open_set = list(sorted(open_set, key=itemgetter(0), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(3)))
            current = open_set[-1]
            # if (current[7] == 0 and len(visited_list) != 0):
            #     open_set = list(sorted(open_set, key=itemgetter(5), reverse=True))
            current = open_set[-1]
            open_set.pop()

            # if find_largest_estimated_reward(visited_list, current[4], current) == 0:
            #     return closed_set, last_ids, visited_list, times
            if current[4] == start and current[2] == termination_time: 
                path = reconstruct_path(closed_set, current)
                printer.print_row(path, save_path, len(visited_list), visited_list, None)
                visited_list.append(path)
                last_ids.append(current[0])
                times.append(time.time() - stime - times[-1])
                print("Path ", len(visited_list), "complete in ", times[-1], "s")
                break
            
            if current[2] != termination_time and within_range(start, current):
                if current[4] != start or current[2] == 0: # only if the start is at time step 0
                    no_children = []
                    for next_node in graph.neighbors(current[4]):
                        reward = graph.cost(current, next_node, closed_set, visited_list)
                        new_reward = current[3] + reward
                        new_t = current[2] + 1
                        if new_reward == 0:
                            no_children.append(False)

                            if debug:
                                    # Debug
                                    branch = reconstruct_path(closed_set, (id, current[0], current[2] + 1, 0, next_node))
                                    printer.print_row(branch, save_path, len(visited_list), visited_list, next_node)
                                    print("")
                            # no children left
                            if current[4] == start and len(no_children) == len(graph.neighbors(current[4])) and not any(no_children):
                                return closed_set, last_ids, visited_list, times
                            else:
                                continue
                        else:
                            no_children.append(True)
                            # new_explored = current[7] + explored
                            if new_t <= termination_time:
                                open_set.append((id, current[0], new_t, new_reward, next_node))
                                closed_set[id] = (current[0], new_t, new_reward, next_node)

                                if debug:
                                    # Debug
                                    branch = reconstruct_path(closed_set, (id, current[0], current[2] + 1, 0, next_node))
                                    printer.print_row(branch, save_path, len(visited_list), visited_list, next_node)
                                    print("")
                        id += 1
                    # branch = reconstruct_path(closed_set, (id, current[0], new_t, new_reward, next_node, reward))
                    # printer.print_row(branch, save_path, len(visited_list), visited_list)
                    # print("")
            
        # if current[4] == start and current[2] == termination_time:
        #     break
    #     if t >= 10:
    #         break
                
    # if t >= 10:
    #     return None, None, open_set
    return closed_set, last_ids, visited_list, times

env = Environment()

# for x in range(WIDTH):
#     for y in range(HEIGHT):
#         env.starting_pos= env.starting_pos._replace(x=x)
#         env.starting_pos =env.starting_pos._replace(y=y)

#         printer = print_results(env.grid, HEIGHT, WIDTH)

#         graph = GridWithWeights(WIDTH, HEIGHT)

#         stime = time.time()
#         # env.starting_pos = Point(0,0)
#         # print(env.grid)

#         closed_set, last_ids, traj = a_star(graph, env.starting_pos, termination_time)
#         t = time.time() - stime
        
#         if closed_set == None:
#             print("Could not complete (%s, %s)" %(str(x), str(y)))
#             file_name = "open_set" + str(x*x+y) + ".txt"
#             np.savetxt(os.path.join(save_path, file_name), traj, fmt='%s')
#             continue
        
#         cost = calc_cost(closed_set, last_ids)
#         path = []
#         for row in traj:
#             for i in row:
#                 path.append((i.x, i.y))
#         print("Termination time: ", termination_time)
#         print("Path:", path)
#         print("Cost:", cost)
#         print("Planning time:", t)

#         printer.print_graph(traj, cost, t, save_path, pos_cnt=x*x+y)

#         plt.close()

env.starting_pos= env.starting_pos._replace(x=0)
env.starting_pos =env.starting_pos._replace(y=0)

printer = print_results(env.grid, HEIGHT, WIDTH)

graph = GridWithWeights(WIDTH, HEIGHT)

stime = time.time()
# env.starting_pos = Point(0,0)
# print(env.grid)

closed_set, last_ids, traj, times = a_star(graph, env.starting_pos, termination_time)
del times[0]

cost = calc_cost(closed_set, last_ids)
path = []
for row in traj:
    for i in row:
        path.append((i.x, i.y))
print("Termination time: ", termination_time)
print("Path:", path)
print("Cost:", cost)
print("Planning time:", sum(times))

if draw:
    printer.print_graph(traj, cost, save_path, times, sum(times), pos_cnt=0)
    plt.close()

file_name = "trajectories.json"
write_json(traj, save_path, file_name)


# Delete any empty folders in PATH
# for folder in os.listdir(PATH):
#     f = os.path.join(PATH, folder)
#     if not len(os.listdir(f)):
#         try:
#             shutil.rmtree(f)
#         except OSError as e:
#             print("Tried to delete folder that doesn't exist: ", f)