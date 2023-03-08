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

termination_time = 8
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
        reachable, distance, heading_home = closest_unvisited(branch, visited_list, current, next_node)
        if next_node in branch or len([True for row in visited_list if next_node in row]) != 0:
            if (not reachable and distance == float('inf')) or (current[3] == 0 and heading_home):
                return None, False, None, None
            # visited
            return 0, reachable, distance, heading_home
        else:
            # unvisited
            return 1, reachable, distance, heading_home

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

    
def closest_unvisited(branch, visited, current, next):
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
    if not bool(distances):
        if dist_next_to_s == float('inf'): return False, None, None
        else: return True, dist_next_to_s, True
    else:
        min_distance =  distances[min(distances, key=distances.get)]#/max_dist
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
    global HH
    visited = 0
    visited_list = []
    closed_set = {}
    closed_set[0] = (-1, 0, 0, start, float('inf'), False)
    id = 1
    open_set = [(0, -1, 0, 0, start, float('inf'), False)]
    current = open_set[-1]
    last_ids = []
    while True:
        printer = print_results(env.grid, HEIGHT, WIDTH)
        # id, parent_id, time, reward, position, visited
        open_set = [(0, 0, 0, 0, start, float('inf'), False)]
        
        t = time.time() - stime
        while len(open_set) > 0:
        # while len(open_set) > 0 and t < 10:
            t = time.time() - stime
            open_set = list(sorted(open_set, key=itemgetter(0), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(3)))
            current = open_set[-1]
            if current[3] == 0 and len(visited_list) != 0 or current[6]:
                open_set = list(sorted(open_set, key=itemgetter(5), reverse=True))
                HH = False
            current = open_set[-1]
            open_set.pop()
            if current[4] == start and current[2] != 0:
                if current[2] == termination_time:
                    if current[3] == 0:
                        break
                    else:
                        path = reconstruct_path(closed_set, current)
                        # printer.print_row(path, save_path, len(visited_list), visited_list)
                        visited_list.append(path)
                        last_ids.append(current[0])
                        print("Path ", len(visited_list), "complete in ", time.time() - stime, "s")
                        break
            
            if current[2] != termination_time and within_range(start, current):
                if current[4] != start or current[2] == 0: # only if the start is at time step 0
                    no_children = []
                    for next_node in graph.neighbors(current[4]):
                        reward, unvisited_reachable, distance, heading_home = graph.cost(current, next_node, closed_set, visited_list)
                        if unvisited_reachable == False:
                            no_children.append(False)
                            # no children left
                            if current[4] == start and len(no_children) == len(graph.neighbors(current[4])) and not all(no_children):
                                continue
                            else:
                                continue
                        else:
                            no_children.append(True)
                            new_reward = current[3] + reward
                            new_t = current[2] + 1
                            if new_t <= termination_time:
                                open_set.append((id, current[0], new_t, new_reward, next_node, distance, heading_home))
                                closed_set[id] = (current[0], new_t, new_reward, next_node, distance, heading_home)

                                # Debug
                                branch = reconstruct_path(closed_set, (id, current[0], current[2] + 1, 0, next_node, 0))
                                printer.print_row(branch, save_path, len(visited_list), visited_list)
                                print("")
                        id += 1
                    # branch = reconstruct_path(closed_set, (id, current[0], new_t, new_reward, next_node, reward))
                    # printer.print_row(branch, save_path, len(visited_list), visited_list)
                    # print("")
            
        if current[3] == 0 and current[2] != 0:
            break
    #     if t >= 10:
    #         break
                
    # if t >= 10:
    #     return None, None, open_set
    return closed_set, last_ids, visited_list

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

closed_set, last_ids, traj = a_star(graph, env.starting_pos, termination_time)
t = time.time() - stime

cost = calc_cost(closed_set, last_ids)
path = []
for row in traj:
    for i in row:
        path.append((i.x, i.y))
print("Termination time: ", termination_time)
print("Path:", path)
print("Cost:", cost)
print("Planning time:", t)

if draw:
    printer.print_graph(traj, cost, t, save_path, pos_cnt=0)
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