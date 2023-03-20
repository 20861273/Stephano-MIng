from operator import itemgetter
import numpy as np

from astar_environment import Environment, HEIGHT, WIDTH, Point
from save_results import print_results

from datetime import datetime
import time
import os
import matplotlib.pyplot as plt
import json
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

import time

import shutil

# termination_time = (WIDTH+HEIGHT)*2
termination_time = 20
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

def draw_tree(closed_set, optimal_trajectories, save_path):
    # path[:-1] returns all the elements of path except the last one
    # path[1:] returns all the elements of path except the first one
    edges = [[(c, p) for p, c in zip(path[:-1], path[1:])] for path in optimal_trajectories]
    
    # Create the graph
    G = nx.DiGraph()
    # fig,ax = plt.subplots()

    # Add the nodes
    for node_id in closed_set:
        parent_id, time_step, reward, position = closed_set[node_id]
        node_label = "node_id\n(%s, %s)" %(str(position.x), str(position.y))
        node = pydot.Node(node_id, label=node_label)
        label_position = "(%s, %s)" %(str(position.x), str(position.y))
        G.add_node(node_id, time_step=time_step, reward=reward, position=label_position)

    # Add the edges
    for node_id in closed_set:
        parent_id, _, _, _ = closed_set[node_id]
        if parent_id != -1:
            # color= ''
            # if (parent_id, node_id) in edges:
            #     color = 'r'
            # else:
            #     color = 'b'
            G.add_edge(parent_id, node_id)#, color=color)
    red_edges = []
    for edge in [e for e in G.edges]:
        for opt_edges in edges:
            if edge in opt_edges:
                red_edges.append(edge)
        
    black_edges = [edge for edge in G.edges() if edge not in red_edges]

    (p,)=nx.drawing.nx_pydot.to_pydot(G)

    # labels = {}
    # for node_id, node_attrs in G.nodes(data=True):
    #     labels[node_id] = f"{node_id}\n{node_attrs['position']}"
    # nx.set_node_attributes(G, labels, 'label')

    # nx.draw_networkx_nodes(G, p, node_size = 600)
    # nx.draw_networkx_labels(G, p, labels, font_size=8)
    # nx.draw_networkx_edges(G, p, edgelist=red_edges, edge_color='r', arrows=True)
    # nx.draw_networkx_edges(G, p, edgelist=black_edges, arrows=False)
    # plt.show()

    file_name = "tree.png"
    file_path = os.path.join(save_path, file_name)

    # p.write_png(file_path)
    graph.write_png(file_path)

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

        # visited
        visited = []
        for row in visited_list:
            for p in row:
                visited.append(p)
        
        visited = visited + branch #+ [next_node]

        estimated_reward = find_largest_estimated_reward(visited, next_node, current)

        # if potential reward < 0
        if estimated_reward == None:
            return None
        elif estimated_reward == float('inf'):
            return float('inf')

        if next_node in branch or len([True for row in visited_list if next_node in row]) != 0:
            return estimated_reward
        else:
            # unvisited
            return 1+estimated_reward

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, left, down, up, stay)
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
    distance = {}
    dist_to_uv = {}
    dist_to_s = {}

    # counts number of unexplored cells within group
    # returns group sizes, position and distance of the closest unexplored cell within each group to next node
    # loop through all the cells in the grid
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # check if the current cell has not been visited and is not marked as visited in the tracker
            if Point(x,y) not in visited and not track_visited[x][y]:
                group_size = 0
                min_distance = float('inf')
                min_dist_to = float('inf')
                min_dist_pos = None
                stack = [(x, y)]
                # depth-first search to find the group of cells connected to the current cell
                while stack:
                    x, y = stack.pop()
                    if not track_visited[x][y]:
                        track_visited[x][y] = True
                        group_size += 1
                        dist_to_uv[Point(x,y)] = get_distance(Point(x,y), next_node)
                        dist_to_s[Point(x,y)] = get_distance(Point(x,y), env.starting_pos)
                        distance[Point(x,y)] = dist_to_uv[Point(x,y)] + dist_to_s[Point(x,y)]

                        if distance[Point(x,y)] <= min_distance and dist_to_uv[Point(x,y)] < min_dist_to:
                            min_dist_to = dist_to_uv[Point(x,y)]
                            min_distance = distance[Point(x,y)]
                            min_dist_pos = Point(x,y)

                    # check the neighbors of the current cell and add them to the stack if they have not been visited
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and Point(nx,ny) not in visited and not track_visited[nx][ny]:
                            stack.append((nx, ny))

                groups.append((group_size, min_dist_pos, min_distance))
    
    # sorts groups from largest to smallest
    groups = sorted(groups, reverse=True)

    # if there aren't any unexplored cells  
    if not bool(groups):
        # checks if agent is at start, then exits
        if current[4] == env.starting_pos:
            return None
        # return home
        return 1 - get_distance(env.starting_pos, next_node)/termination_time # changed from None to get exit condition for area covered
    
    PR = 0
    for i, group in enumerate(groups):
        group_max_PR = 0
        # potential reward calculation
        remainder_steps = termination_time - group[2] - current[2]
        # can't reach
        if remainder_steps < 0:
            continue

        # if the size of the group is smaller than the remainder steps the potential reward can only be as large as the gorup
        if group[0] > remainder_steps:
            group_max_PR = remainder_steps
        else:
            group_max_PR = group[0]

        if group_max_PR > PR:
            PR = group_max_PR
            # TODO
            # check if the next group can be reached
            # if len(groups) > 1 and i != len(groups)-1:
            #     temp = groups[i+1][2] - remainder_steps
            #     if temp 
        
    
    # no potential reward
    if PR < 0:
        return None
    return PR/termination_time

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
    rememeber_ids = []
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
            open_set.pop()

            if current[4] == start and current[2] == termination_time: 
                path = reconstruct_path(closed_set, current)
                if debug: printer.print_row(path, save_path, len(visited_list), visited_list, None)
                visited_list.append(path)
                last_ids.append(current[0])
                times.append(time.time() - stime - times[-1])
                print("Path ", len(visited_list), "complete in ", times[-1], "s")
                break
            
            if current[2] != termination_time and within_range(start, current):
                no_children = []
                no_reward = []
                for next_node in graph.neighbors(current[4]):
                    if debug:
                        # Debug
                        branch = reconstruct_path(closed_set, (id, current[0], current[2] + 1, 0, next_node))
                        printer.print_row(branch, save_path, len(visited_list), visited_list, next_node)
                        print("")
                    reward = graph.cost(current, next_node, closed_set, visited_list)
                    if reward == 0:
                        rememeber_ids.append(id)
                        no_reward.append(False)
                    if reward == None:
                        breakpoint
                    if reward != None: new_reward = current[3] + reward
                    # no path or all children got no reward or 
                    if reward == None or (len(no_reward) == len(graph.neighbors(current[4])) and not any(no_reward)) or reward == float('inf'):
                        no_children.append(False)
                        # no children left
                        if (current[4] == start and len(no_children) == len(graph.neighbors(current[4])) and not any(no_children)) \
                            or (len(no_reward) == len(graph.neighbors(current[4])) and not any(no_reward) and current[4] == env.starting_pos)\
                                    or reward == float('inf'):
                            return closed_set, last_ids, visited_list, times
                        else:
                            breakpoint
                            continue
                    else:
                        no_children.append(True)
                        new_t = current[2] + 1
                        # new_explored = current[7] + explored
                        if new_t <= termination_time:
                            open_set.append((id, current[0], new_t, new_reward, next_node))
                            closed_set[id] = (current[0], new_t, new_reward, next_node)
                    id += 1
            
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

cost, paths_ids = calc_cost(closed_set, last_ids)
path = []
for row in traj:
    for i in row:
        path.append((i.x, i.y))
print("Termination time: ", termination_time)
print("Path:", path)
print("Cost:", cost)
print("Planning time:", sum(times))

# draw_tree(closed_set, paths_ids, save_path)

if draw:
    printer.print_graph(traj, cost, save_path, times, sum(times), pos_cnt=0)
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