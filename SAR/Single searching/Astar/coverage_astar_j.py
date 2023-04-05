from operator import itemgetter
import numpy as np

from astar_environment import Environment, HEIGHT, WIDTH, Point, States
from save_results import print_results

from datetime import datetime
import time
import os
import matplotlib.pyplot as plt
import json

import time

import shutil

# termination_time = (WIDTH+HEIGHT)*2
termination_time = 15
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
        self.explored_cells = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.weights

    def collision(self, id):
        (x, y) = id
        return env.grid[y][x] != States.OBS.value

    def cost(self, current, next_node, closed_set, visited_list):
        branch = reconstruct_path(closed_set, current)

        # visited
        visited = []
        for row in visited_list:
            for p in row:
                visited.append(p)
        
        visited = visited + branch

        reachable, area_covered = closest_unvisited(visited, current, next_node)

        if (not reachable) and current[3] < 1:
            breakpoint
            return None
        if area_covered:
            breakpoint

        if current[3] < 1 or area_covered:
            if next_node not in branch:
                if len([True for row in visited_list if next_node in row]) == 0:
                    return 1
                else:
                    return 0.00000000000001
            else:
                return 0
        else:
            if next_node not in visited:
                return 1
            else:
                return 0
        
        breakpoint

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, left, down, up, stay)
        results = [Point(x+1, y), Point(x-1, y), Point(x, y-1), Point(x, y+1), Point(x, y)]
        # This is done to prioritise straight paths
        # if (x + y) % 2 == 0: results = [Point(x-1, y), Point(x+1, y), Point(x, y-1), Point(x, y+1), Point(x, y)]
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.collision, results))
        return results

def closest_unvisited(visited, current):
    distances = {}
    # gets the distance to all unvisited blocks within reach
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if Point(x,y) not in visited:
                dist_to_u = get_distance(Point(x,y), current[6])
                dist_to_s = get_distance(Point(x,y), env.starting_pos)
                distance = dist_to_u + dist_to_s

                # distance = get_distance(Point(x,y), current[6])

                if distance <= current[2]:
                    distances[Point(x,y)] = distance
    
    # Check if unexplored blocks reachable
    if not bool(distances):
        return None, 0
    else:
        return min(distances, key=distances.get), len(distances)

def get_distance(end, start):
    return abs(start.x - end.x) + abs(start.y - end.y)

def within_range(start, current):
    dist = abs(current[4].x - start.x) + abs(current[4].y - start.y)
    if dist <= termination_time-current[2]:
        return True
    return False

def reconstruct_path(closed_set, current):
    current = current[1:]
    path = [current[5]]
    while current[0] != -1:
        current = closed_set[current[0]]
        path.append(current[5])
    path.reverse()
    return path

def calc_cost(closed_set, ids):
    cost = 0
    paths = []
    for id in ids:
        costs = [closed_set[id]]
        current = closed_set[id]
        path_ids = [closed_set[id][0]]
        while True:
            if current[0] == -1: break
            current = closed_set[current[0]]
            costs.append(current)
            path_ids.append(current[0])
        paths.append(path_ids)
        for c in costs: 
            if c[3] == closed_set[0][3]: cost += c[2]
    return cost, paths

def check_unexplored(next_node, explored):
    if next_node in explored:
        return 0
    else:
        return 1

# A* algorithm
def a_star(graph, start, termination_time):
    times = [0]
    debug = False
    debug_found = False
    visited_list = []
    closed_set = {}
    id = 0
    open_set = []
    last_ids = []
    explored = []
    explored.append(start)
    while True:
        printer = print_results(env.grid, HEIGHT, WIDTH)
        # 0: id, 1: parent_id, 2: fuel, 3: score_to_come, 4: score_to_go, 5: total_score, 6: position
        open_set = [(0, -1, termination_time, 0, 0, 0, start)]
        current = list(open_set[-1])
        
        for row in visited_list:
            for p in row:
                explored.append(p)
        
        closest_unexplored, reachable_unexplored_cells = closest_unvisited(explored, current)
        if closest_unexplored == None:
            return closed_set, last_ids, visited_list, times

        if current[2] >= get_distance(current[6], closest_unexplored)*2:
            current[4] = 1
        else:
            return closed_set, last_ids, visited_list, times
        
        current[5] = current[3] + current[4]
        current[0] = 0
        current[1] = -1
        closed_set[current[0]] = (tuple(current[1:]))
        open_set.append(current)
        
        while len(open_set) > 0:
            current = open_set[-1]
            open_set.pop()

            branch_from_current = reconstruct_path(closed_set, current)
            explored_from_current = explored+branch_from_current

            # if used all fuel and drone is back at initial position, then goal reached (allow a margin of 1 fuel for even/odd number of moves)
            # if current[2] <= 1 and current[6] == start:
            # if no more reachable unexplored cells and drone is back at initial position, then goal reached
            if (current[6] == start) \
            and (current[2] < get_distance(current[6], closest_unexplored) + get_distance(closest_unexplored, current[6])):
                # reconstruct best coverage path by starting at last state and iterating backwards through parents states
                path = reconstruct_path(closed_set, current)
                # Debug purposes
                if debug_found: printer.print_row(path, save_path, len(visited_list), visited_list, None, env)
                visited_list.append(path)
                last_ids.append(current[0])
                times.append(time.time() - stime - times[-1])
                print("Path ", len(visited_list), "complete in ", times[-1], "s")
                break

            # otherwise, generate child states        
            for next_node in graph.neighbors(current[6]):
                # Debuggin purposes
                if debug:
                    branch = reconstruct_path(closed_set, (id, current[0], current[2] + 1, 0, 0, 0, next_node))
                    printer.print_row(branch, save_path, len(visited_list), visited_list, next_node, env)
                    breakpoint
                
                next_fuel = current[2] - 1

                # check whether there is enough fuel left to get back to the start position from the next state
                # if the initial position cannot be reached from the next state, then do not add the next state to the tree or the priority queue
                # if the initial condition can be reached from the next state, then calculate the score of the next state and add it to the tree and the priority queue
                if next_fuel < get_distance(next_node, start):
                    # do not add next_state to tree or priority queue
                    continue
                else:
                    # the score to come is the score to come so far plus the unexplored status (1 or 0) of the next state
                    next_score_to_come = current[3] + check_unexplored(next_node, explored_from_current)

                # the next state's score to go is 1 if there is any unexplored cell that the drone can reach and also return to the initial position afterwards
                # the next state's score to go is 0 if there are no unexplored cells within reach, but the drone can still reach the initial position
                # if next_fuel >= get_distance(next_node, closest_unexplored) + get_distance(closest_unexplored, start):
                #     next_score_to_go   = 1
                # elif next_fuel >= get_distance(next_node, start):
                #     next_score_to_go   = 0

                # the score to go is the number of unexplored cells that can be reached and also reach the initial position
                closest_unexplored_from_next, reachable_unexplored_cells_from_next = closest_unvisited(explored_from_current, (0, 0, next_fuel, 0, 0, 0, next_node))
                next_score_to_go = reachable_unexplored_cells_from_next 

                # the total score is the sum of the score to come and the score to go
                next_score = next_score_to_come + next_score_to_go

                # increment the state ID, record the parent state ID, and add the next state to the tree and the priority queue
                id = id + 1
                next_id = id
                next_parentID = current[0]
                next_state = (next_id, next_parentID, next_fuel, next_score_to_come, next_score_to_go, next_score, next_node)
                closed_set[id] = (next_state[1:])
                open_set.append(next_state)
                # explored.append(next_node)
            # after generating all the child states from the current state, sort the priority queue from highest to lowest score
            open_set = list(sorted(open_set, key=itemgetter(0), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(5)))
        # if the priority queue is empty and the initial node was never reached after using all the fuel, then return empty path
        

env = Environment()

env.starting_pos= env.starting_pos._replace(x=0)
env.starting_pos =env.starting_pos._replace(y=0)

printer = print_results(env.grid, HEIGHT, WIDTH)

graph = GridWithWeights(WIDTH, HEIGHT)

stime = time.time()
# env.starting_pos = Point(0,0)
# print(env.grid)

closed_set, last_ids, traj, times = a_star(graph, env.starting_pos, termination_time)
del times[0]

cost, paths_ids = calc_cost(closed_set, last_ids)
path = []
for row in traj:
    for i in row:
        path.append((i.x, i.y))
print("Termination time: ", termination_time)
print("Path:", path)
print("Cost:", cost)
print("Planning time:", sum(times))

if draw:
    printer.print_graph(traj, cost, save_path, times, sum(times), 0, env)
    plt.close()

file_name = "trajectories.json"
write_json(traj, save_path, file_name)
file_name = "closed_set.json"
file_path = os.path.join(save_path, file_name)
with open(file_path, 'w') as f:
    for key, value in closed_set.items():
        json.dump({key: value}, f)
        f.write('\n')


# Delete any empty folders in PATH
for folder in os.listdir(PATH):
    f = os.path.join(PATH, folder)
    if not len(os.listdir(f)):
        try:
            shutil.rmtree(f)
        except OSError as e:
            print("Tried to delete folder that doesn't exist: ", f)