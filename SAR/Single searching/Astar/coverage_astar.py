from operator import itemgetter

from astar_environment import Environment, HEIGHT, WIDTH, Point
from save_results import print_results

from datetime import datetime
import time
import os
import matplotlib.pyplot as plt

import shutil

termination_time = 4

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

    def passable(self, id):
        return id not in self.weights

    def cost(self, current, next_node, closed_set, visited_list):
        branch = reconstruct_path(closed_set, current)
        closest_unvisted = self.closest_unvisted(branch, visited_list, current[4])
        if next_node in branch or len([True for row in visited_list if next_node in row]) != 0:
            # visited
            # return -1, closest_unvisted #here
            return 0, closest_unvisted
        else:
            # unvisited
            return 1, closest_unvisted

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x-1, y), Point(x, y-1), Point(x, y+1), Point(x, y)]
        # This is done to prioritise straight paths
        # if (x + y) % 2 == 0: results.reverse()
        results = filter(self.in_bounds, results)
        # results = filter(self.passable, results)
        return results

    def closest_unvisted(self, branch, visited, current):
        grid = env.grid.copy()
        grid.fill(0)
        distances = {}
        for row in visited:
            for p in row:
                grid[p.y, p.x] = 1
        for row in visited:
            for p in row:
                grid[p.y, p.x] = 1
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y,x] == 0: distances[Point(y,x)] = self.get_distance(Point(y,x), current)
        
        max_dist = self.get_distance(Point(0,0), Point(HEIGHT-1,WIDTH-1))

        return distances[min(distances, key=distances.get)]/max_dist
    
    def get_distance(self, unvisited, current):
        return abs(current.x - unvisited.x) + abs(current.y - unvisited.y)

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

def within_range(start, current):
    dist = abs(current[4].x - start.x) + abs(current[4].y - start.y)
    if dist <= termination_time-current[2]:
        return True
    return False

# A* algorithm
def a_star(graph, start, termination_time):
    visited = 0
    visited_list = []
    closed_set = {}
    closed_set[0] = (-1, 0, 0, start)
    id = 1
    open_set = [(0, -1, 0, 0, start)]
    current = open_set[-1]
    last_ids = []
    while True:
        printer = print_results(env.grid, WIDTH, HEIGHT)
        # if id != 1: open_set = [(current[0], current[1], 0, 0, current[4])] # id, parent_id, time, reward, position, visited
        open_set = [(0, 0, 0, 0, start)]
        
        while len(open_set) > 0:
            open_set = list(sorted(open_set, key=itemgetter(0), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(3)))
            current = open_set[-1]
            open_set.pop()
            if current[4] == start and current[2] != 0:
                if current[2] == termination_time:
                    # if current[3] <= (termination_time-1)*-1: #here
                    if current[3] == 0:
                        break
                    else:
                        # branch = reconstruct_path(closed_set, current)
                        # printer.print_row(branch, save_path, id)
                        # print("")
                        path = reconstruct_path(closed_set, current)
                        if len(visited_list) == 1: del path[0]
                        visited_list.append(path)
                        last_ids.append(current[0])
                        break
            
            if current[2] != termination_time and within_range(start, current):
                if current[4] != start or current[2] == 0: # only if the start is at time step 0
                    for next_node in graph.neighbors(current[4]):
                        reward, dis = graph.cost(current, next_node, closed_set, visited_list)
                        new_reward = current[3] + reward #+ dis #here
                        new_t = current[2] + 1
                        if new_t <= termination_time:
                            open_set.append((id, current[0], new_t, new_reward, next_node))
                            closed_set[id] = (current[0], new_t, new_reward, next_node)

                            # Debug
                            # branch = reconstruct_path(closed_set, (id, current[0], new_t, new_reward, next_node, reward))
                            # printer.print_graph(branch, HEIGHT, WIDTH)
                            # print("")
                        id += 1
            
        # if current[3] <= (termination_time-1)*-1: #here
        if current[3] == 0:
            break
                
                    
    return closed_set, last_ids, visited_list

env = Environment()
printer = print_results(env.grid, WIDTH, HEIGHT)

graph = GridWithWeights(WIDTH, HEIGHT)

stime = time.time()

closed_set, last_ids, traj = a_star(graph, Point(2,2), termination_time)
t = time.time() - stime
cost = calc_cost(closed_set, last_ids)
path = []
for row in traj:
    for i in row:
        path.append((i.x, i.y))
print("Path:", path)
print("Cost:", cost)

printer.print_graph(traj, cost, t, save_path)

plt.close()

# Delete any empty folders in PATH
for folder in os.listdir(PATH):
    f = os.path.join(PATH, folder)
    if not len(os.listdir(f)):
        try:
            shutil.rmtree(f)
        except OSError as e:
            print("Tried to delete folder that doesn't exist: ", f)