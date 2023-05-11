from operator import itemgetter
import numpy as np
from math import trunc

from astar_environment import Environment, HEIGHT, WIDTH, Point, States
from save_results import print_results

from datetime import datetime
import time
import os
import matplotlib.pyplot as plt
import json
import heapq
import networkx as nx
from EoN import hierarchy_pos

import time

import shutil

# termination_time = (WIDTH+HEIGHT)*2
termination_time = 10
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

def make_dag(closed_set):
    # Create a directed graph
    G = nx.DiGraph()

    # Add the nodes to the graph
    for child_id, (parent_id, _, _, _, priority, position) in closed_set.items():
        G.add_node(child_id, priority=priority)

    # Add the edges to the graph
    for child_id, (parent_id, _, _, _, priority, position) in closed_set.items():
        if parent_id is not None:
            G.add_edge(parent_id, child_id)

    # Initialize the heap list with the root node
    # heap = ["%s\n%s,%s" %(str(0), str(closed_set[0][5].x), str(closed_set[0][5].y))]
    heap = [(0, (closed_set[0][5].x, closed_set[0][5].y))]

    # Loop through the rest of the nodes and add them to the heap
    for key in closed_set:
        if key == 0: continue
        # heapq.heappush(heap, ("%s\n%s,%s" %(str(key), str(closed_set[key][5].x), str(closed_set[key][5].y))))
        heapq.heappush(heap, (key, (closed_set[key][5].x, closed_set[key][5].y)))
        
    return G, heap

def draw_tree(closed_set, cycle_num, dir_path, ax):
    G, heap = make_dag(closed_set)
    pos = hierarchy_pos(G, width=0.1)
    labels = dict(enumerate(heap))
    if ax == None: nx.draw(G, pos, node_size=2000, labels=labels, alpha=0.4, font_size=14, ax=ax)
    else: nx.draw(G, pos, node_size=2000, labels=labels, alpha=0.4, font_size=14, ax=ax)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18)
    file_name = "tree%s.png"%(str(cycle_num))
    plt.savefig(os.path.join(dir_path, file_name), bbox_inches='tight')
    # plt.pause(.1)
    # breakpoint
    plt.close()

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
        self.explored_cells = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def collision(self, id):
        (x, y) = id
        return env.grid[y][x] != States.OBS.value

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
    distances_to_next = {}
    distances_to_start = {}
    reachable_unexplored_cells= []
    # gets the distance to all unvisited blocks within reach
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if Point(x,y) not in visited:
                dist_to_u = get_distance(Point(x,y), current[6])
                dist_to_s = get_distance(Point(x,y), env.starting_pos)
                distance = dist_to_u + dist_to_s

                # distance = get_distance(Point(x,y), current[6])

                if distance <= current[2]:
                    distances_to_next[Point(x,y)] = dist_to_u
                    distances_to_start[Point(x,y)] = dist_to_s
                    reachable_unexplored_cells.append(Point(x,y))
    
    # Check if unexplored blocks reachable
    if not bool(distances_to_next):
        return None, None, reachable_unexplored_cells
    else:
        return min(distances_to_next, key=distances_to_next.get), min(distances_to_start, key=distances_to_start.get), reachable_unexplored_cells

def get_distance(end, start):
    return abs(start.x - end.x) + abs(start.y - end.y)

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
    
def set_explored_cells(explored):
    for cell in explored:
        if cell not in graph.explored_cells:
            graph.explored_cells.append(cell)

def get_explored_so_far(branch):
    explored_so_far = 0
    if len(graph.explored_cells) == 0: return explored_so_far
    for cell in branch:
        if cell not in graph.explored_cells:
            explored_so_far += 1
    return explored_so_far

# A* algorithm
def a_star(graph, start, termination_time):
    times = [0]
    debug = False
    debug_found = False
    save_data = False
    visited_list = []
    closed_set = {}
    id = 0
    open_set = []
    last_ids = []
    explored = []
    cycle_closed_set = {}
    explored.append(start)
    while True:
        # if len(cycle_closed_set) != 0:
        #     draw_tree(cycle_closed_set, len(visited_list), save_path, ax=None)
        cycle_closed_set = {}
        printer = print_results(env.grid, HEIGHT, WIDTH)
        # 0: id, 1: parent_id, 2: fuel, 3: score_to_come, 4: score_to_go, 5: total_score, 6: position
        open_set = [(0, -1, termination_time, 0, 0, 0, start)]
        current = list(open_set[-1])
        
        for row in visited_list:
            for p in row:
                explored.append(p)
        
        closest_unexplored_from_current, closest_unexplored_from_start, reachable_unexplored_cells = closest_unvisited(explored, current)
        if closest_unexplored_from_current == None:
            return closed_set, last_ids, visited_list, times

        if current[2] >= get_distance(current[6], closest_unexplored_from_current)*2:
            current[4] = 1
        else:
            return closed_set, last_ids, visited_list, times
        
        current[5] = current[3] + current[4]
        current[0] = 0
        current[1] = -1
        closed_set[current[0]] = (tuple(current[1:]))
        cycle_closed_set[current[0]] = (tuple(current[1:]))
        open_set.append(current)
        set_explored_cells(explored)
        
        while len(open_set) > 0:
            current = open_set[-1]
            open_set.pop()

            branch_from_current = reconstruct_path(closed_set, current)
            explored_from_current = explored+branch_from_current
            closest_unexplored_from_current, closest_unexplored_from_start, reachable_unexplored_cells = closest_unvisited(explored, current)

            # if used all fuel and drone is back at initial position, then goal reached (allow a margin of 1 fuel for even/odd number of moves)
            # if current[2] <= 1 and current[6] == start:
            # if no more reachable unexplored cells and drone is back at initial position, then goal reached
            if (current[6] == start) \
            and len(reachable_unexplored_cells) == 0:
                # reconstruct best coverage path by starting at last state and iterating backwards through parents states
                path = reconstruct_path(closed_set, current)
                # Debug purposes
                if debug_found: printer.print_row(path, save_path, len(visited_list), visited_list, None, env)
                visited_list.append(path)
                last_ids.append(current[0])
                times.append(time.time() - stime - times[-1])
                min, s = divmod(times[-1], 60)
                print("Path ", len(visited_list), "complete in ", round(min,2), "m ", round(s,2), "s")
                break

            # otherwise, generate child states        
            for next_node in graph.neighbors(current[6]):
                # Debuggin purposes
                if debug:
                    branch = reconstruct_path(closed_set, (id, current[0], current[2] + 1, 0, 0, 0, next_node))
                    printer.print_row(branch, save_path, len(visited_list), visited_list, next_node, env)
                    breakpoint                   
                
                next_fuel = current[2] - 1
                explored_so_far = get_explored_so_far(branch_from_current)

                # check whether there is enough fuel left to get back to the start position from the next state
                # if the initial position cannot be reached from the next state, then do not add the next state to the tree or the priority queue
                # if the initial condition can be reached from the next state, then calculate the score of the next state and add it to the tree and the priority queue
                if next_fuel < get_distance(next_node, start):
                    # do not add next_state to tree or priority queue
                    continue

                closest_unexplored_from_next, closest_unexplored_from_start, reachable_unexplored_cells = closest_unvisited(explored_from_current, (0, 0, next_fuel, 0, 0, 0, next_node))
                
                # if the next state has not been explored add 1 to explored so far
                if next_node not in explored_from_current:
                    explored_so_far += 1
                elif len(reachable_unexplored_cells) == 0:
                    breakpoint
                
                if len(reachable_unexplored_cells) == 0:
                    # do not add next_state to tree or priority queue
                    next_score =    (                                                       \
                                    explored_so_far                                         \
                                    +len(reachable_unexplored_cells)                        \
                                    )                                                       \
                                    /                                                       \
                                    (                                                       \
                                    termination_time-next_fuel                              \
                                    +get_distance(start, next_node)                         \
                                    )
                else:
# next_score =
#                              unexplored cells explored so far + number of reachable cells from here
#               -------------------------------------------------------------------------------------------------
#(distance travelled so far) + (distance between current position and closest unexplored cell) + (distance betweem home position and reachable unexplored cell closet to it)
                
                    next_score =    (                                                       \
                                    explored_so_far                                         \
                                    +len(reachable_unexplored_cells)              \
                                    )                                                       \
                                    /                                                       \
                                    (                                                       \
                                    termination_time-next_fuel                              \
                                    +get_distance(next_node, closest_unexplored_from_next)  \
                                    +(len(reachable_unexplored_cells)-1)                         \
                                    +get_distance(start, closest_unexplored_from_start)     \
                                    )

                # increment the state ID, record the parent state ID, and add the next state to the tree and the priority queue
                id = id + 1
                next_id = id
                next_parentID = current[0]
                next_state = (next_id, next_parentID, next_fuel, 0, 0, next_score, next_node)
                closed_set[id] = (next_state[1:])
                cycle_closed_set[id] = (next_state[1:])
                open_set.append(next_state)
                
                if save_data:
                    file_name = "closed_set%d.json" %(next_id)
                    file_path = os.path.join(save_path, file_name)
                    with open(file_path, 'w') as f:
                        for key, value in closed_set.items():
                            json.dump({key: value}, f)
                            f.write('\n')
                    file_name = "open_set%d.json" %(next_id)
                    file_path = os.path.join(save_path, file_name)
                    with open(file_path, 'w') as f:
                        for state in open_set:
                            json.dump(state, f)
                            f.write('\n')
                
            # after generating all the child states from the current state, sort the priority queue from highest to lowest score
            open_set = list(sorted(open_set, key=itemgetter(0), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(5)))

        # if the priority queue is empty and the initial node was never reached after using all the fuel, then return empty path
        

env = Environment()

env.starting_pos= env.starting_pos._replace(x=50)
env.starting_pos =env.starting_pos._replace(y=50)

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