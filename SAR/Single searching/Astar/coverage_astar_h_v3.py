from operator import itemgetter
import numpy as np

from astar_environment import Environment, HEIGHT, WIDTH, Point, States
from save_results import print_results

from datetime import datetime
import time
import os
import matplotlib.pyplot as plt

import time

import shutil

termination_time = 6
MAX_DISTANCE = (HEIGHT+WIDTH-2) * 2

PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'Astar')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path)
date_and_time = datetime.now()
save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
if not os.path.exists(save_path): os.makedirs(save_path)

class GridWithWeights(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.weights = {}
        self.visited_blocks = set()

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def collision(self, id):
        (x, y) = id
        return env.grid[y][x] != States.OBS.value

    def passable(self, id):
        return id not in self.weights

    def cost(self, current, next_node, closed_set, visited_list):
        branch = reconstruct_path(closed_set, current)
        # visited
        visited = []
        for row in visited_list:
            for p in row:
                visited.append(p)
        
        visited = visited + branch
        reachable, reward = closest_unvisited(visited, branch, current, next_node)

        return reward, reachable
        
        # if next_node in visited:
        #     # visited
        #     return distance, reachable
        # else:
        #     # unvisited
        #     return distance*2, reachable

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x-1, y), Point(x, y-1), Point(x, y+1), Point(x, y)]
        # This is done to prioritise straight paths
        # if (x + y) % 2 == 0: results.reverse()
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.collision, results))
        return results

def within_range(start, current):
    dist = abs(current[4].x - start.x) + abs(current[4].y - start.y)
    if dist <= termination_time-current[2]:
        return True
    return False

    
def closest_unvisited(visited, branch, current, next):
    distances = {}
    remaining_time = termination_time - current[2]
    # gets the distance to all unvisited blocks within reach
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if Point(x,y) not in visited:
                dist_to_u = get_distance(Point(x,y), next)
                dist_to_s = get_distance(Point(x,y), env.starting_pos)
                distance = dist_to_u + dist_to_s

                if distance < remaining_time:
                    distances[Point(x,y)] = distance

    # Check if unexplored blocks reachable
    if not bool(distances):
        # if agent at start, then end run
        if current[4] == env.starting_pos: return False, None
        else:
            reachable = within_range(env.starting_pos, (None, None, current[2]+1, None, next))
            if current[3] > 0 and reachable:
                return True, -0.0001 # agent cannot reah unexplored cell, but can reach start after it has explored an unexplored cell
            else:
                return False, 0 # agent cannot reach unexplored cell or start
    else:
        if next in branch:
            return True, 0.5  # agent can reach unexplored cell and start, but next node has been explored this run
        elif next in visited:
            return True, 1 # agent can reach unexplored cell and start, but next node has been explored in previous runs
        else:
            return True, 2 # agent can reach unexplored cell and start, but next node is unexplored
    

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
    visited = 0
    debug = True
    debug_found = True
    visited_list = []
    closed_set = {}
    closed_set[0] = (-1, 0, 0, start)
    id = 1
    open_set = [(0, -1, 0, 0, start)]
    current = open_set[-1]
    last_ids = []
    while True:
        printer = print_results(env.grid, HEIGHT, WIDTH)
        # if id != 1: open_set = [(current[0], current[1], 0, 0, current[4])] # id, parent_id, time, reward, position, visited
        open_set = [(0, 0, 0, 0, start)]
        
        while len(open_set) > 0:
            # open_set = list(sorted(open_set, key=itemgetter(5), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(0), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(3)))
            current = open_set[-1]
            open_set.pop()
            if current[4] == start and current[2] == termination_time:
                path = reconstruct_path(closed_set, current)
                if debug_found: printer.print_row(path, save_path, len(visited_list), visited_list, None, env)
                # if len(visited_list) == 1: del path[0]
                visited_list.append(path)
                last_ids.append(current[0])
                times.append(time.time() - stime - times[-1])
                print("Path ", len(visited_list), "complete in ", times[-1], "s")
                break
            
            if current[2] < termination_time and within_range(start, current):
                no_children = []
                for next_node in graph.neighbors(current[4]):
                    if debug:
                        # Debug
                        branch = reconstruct_path(closed_set, (id, current[0], current[2] + 1, 0, next_node, 0))
                        printer.print_row(branch, save_path, len(visited_list), visited_list, next_node, env)
                        breakpoint
                    reward, unvisited_reachable = graph.cost(current, next_node, closed_set, visited_list)
                    if not unvisited_reachable:
                        no_children.append(False)
                        # no children left
                        if current[2] == 0 and len(no_children) == len(graph.neighbors(current[4])) and not any(no_children):
                            return closed_set, last_ids, visited_list, times
                        breakpoint
                        continue
                    else:
                        no_children.append(True)
                        new_reward = current[3] + reward
                        new_t = current[2] + 1
                        open_set.append((id, current[0], new_t, new_reward, next_node))
                        closed_set[id] = (current[0], new_t, new_reward, next_node)
                            
                    id += 1
            
        # if current[3] <= (termination_time-1)*-1: #here
        if current[3] == 0 and current[2] != 0:
            break
                
                    
    return closed_set, last_ids, visited_list, times

env = Environment()
printer = print_results(env.grid, HEIGHT, WIDTH)

graph = GridWithWeights(WIDTH, HEIGHT)

stime = time.time()
env.starting_pos = Point(0,0)
print(env.grid)

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

printer.print_graph(traj, cost, save_path, times, sum(times), 0, env)

plt.close()

# Delete any empty folders in PATH
# for folder in os.listdir(PATH):
#     f = os.path.join(PATH, folder)
#     if not len(os.listdir(f)):
#         try:
#             shutil.rmtree(f)
#         except OSError as e:
#             print("Tried to delete folder that doesn't exist: ", f)