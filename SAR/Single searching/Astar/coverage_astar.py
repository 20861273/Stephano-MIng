from operator import itemgetter

from astar_environment import Environment, HEIGHT, WIDTH, Point
from save_results import print_results

from datetime import datetime
import os
import matplotlib.pyplot as plt

termination_time = 300
visited_cutoff = 20

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
        if next_node in branch or next_node in visited_list:
            # has visited
            return 0
            # End checker. Checks for termination condition
            if next_node == closed_set[0][4] and current[2] != termination_time:
                return 0, 0
            else:
                return 0, 1
        else:
            return 1
            # not visited
            if next_node == closed_set[0][4] and current[2] != termination_time:
                return 1, 0
            else:
                return 1, 1

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x-1, y), Point(x, y-1), Point(x, y+1), Point(x, y)]
        # This is done to prioritise straight paths
        # if (x + y) % 2 == 0: results.reverse()
        results = filter(self.in_bounds, results)
        # results = filter(self.passable, results)
        return results

def reconstruct_path(closed_set, current):
    current = current[1:5]
    path = [current[3]]
    while current[0] != -1:
        current = closed_set[current[0]]
        path.append(current[3])
    path.reverse()
    return path

def within_range(start, current):
    dist = abs(current[4].x - start.x) + abs(current[4].y - start.y)
    if dist <= termination_time-current[2]:
        return True
    return False

# A* algorithm
def a_star(graph, start, termination_time, visited_cutoff):
    visited = 0
    visited_list = []
    closed_set = {}
    closed_set[0] = (-1, 0, 0, start)
    id = 1
    open_set = [(0, -1, 0, 0, start)]
    current = open_set[-1]
    while visited < visited_cutoff:
        printer = print_results(env.grid, WIDTH, HEIGHT)
        # if id != 1: open_set = [(current[0], current[1], 0, 0, current[4])] # id, parent_id, time, reward, position, visited
        open_set = [(0, -1, 0, 0, start)]
        
        while len(open_set) > 0:
            open_set = list(sorted(open_set, key=itemgetter(0), reverse=True))
            open_set = list(sorted(open_set, key=itemgetter(3)))
            current = open_set[-1]
            open_set.pop()
            if current[4] == start and current[2] != 0:
                if current[2] == termination_time:
                    # branch = reconstruct_path(closed_set, current)
                    # printer.print_graph(branch, HEIGHT, WIDTH)
                    # plt.show()
                    # print("")
                    # plt.close()
                    visited = 0
                    path = reconstruct_path(closed_set, current)
                    if len(visited_list) != 0: del path[0]
                    for i in path:
                        if i in visited_list: visited += 1
                    if visited < visited_cutoff:
                        visited_list = visited_list + path

                    break
            
            if current[2] != termination_time and within_range(start, current):
                if current[4] != start or current[2] == 0: # only if the start is at time step 0
                    for next_node in graph.neighbors(current[4]):
                        reward = graph.cost(current, next_node, closed_set, visited_list)
                        new_reward = current[3] + reward
                        new_t = current[2] + 1
                        if new_reward <= termination_time and new_t <= termination_time:
                            open_set.append((id, current[0], new_t, new_reward, next_node))
                            closed_set[id] = (current[0], new_t, new_reward, next_node)

                            # Debug
                            # branch = reconstruct_path(closed_set, (id, current[0], new_t, new_reward, next_node, reward))
                            # printer.print_graph(branch, HEIGHT, WIDTH)
                            # print("")
                        id += 1
                
                    
    return closed_set, current, visited_list

env = Environment()
printer = print_results(env.grid, WIDTH, HEIGHT)

graph = GridWithWeights(WIDTH, HEIGHT)

closed_set, current, traj = a_star(graph, Point(2,2), termination_time, visited_cutoff)
print("Path:", [(i.x, i.y) for i in traj])
print("Cost:", closed_set[current[0]][2])

printer.print_graph(traj, HEIGHT, WIDTH)
file_name = "traj.png"
plt.savefig(os.path.join(save_path, file_name))
plt.close()