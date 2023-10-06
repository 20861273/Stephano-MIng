import heapq
from collections import namedtuple
import collections
from enum import Enum
import numpy as np
from enclosed_space_checker import Enclosed_space_check
import matplotlib.pyplot as plt
import os
from datetime import datetime

Point = namedtuple('Point', 'x, y')
HEIGHT = 6
WIDTH = 6
class States(Enum):
    UNEXP = 0
    OBS = 1
    ROBOT = 2
    GOAL = 3
    EXP = 4
class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class GridWithWeights(object):
    def __init__(self, height, width, grid): ########################################### change
        self.grid = grid
        self.width = width
        self.height = height

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def is_unoccupied(self, id):
        (x, y) = id
        return self.grid[y,x] != States.OBS.value

    def neighbors(self, id, step=None, occupied_cells=None):
        (x, y) = id
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        # This is done to prioritise straight paths
        #if (x + y) % 2 == 0: results.reverse()
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.is_unoccupied, results))
        return results

class Astar:
    def __init__(self, height, width, grid):
        self.came_from = {}
        self.cost_so_far = {}
        self.grid = grid
        self.graph = GridWithWeights(height, width, grid)

    # Heuristic function to estimate the cost from a current block to a end block
    def heuristic(self, current, end):
        return abs(current[0] - end.x) + abs(current[1] - end.y)
    
    # Path reconstruction
    def reconstruct_path(self, start, end): #: dict[Location, Location], : Location, : Location
        current = end
        path = [current]
        while current != start:
            if current not in self.came_from: print(self.grid, self.start, end)
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path

    # A* algorithm
    def a_star(self, start, end):
        self.start = start
        self.came_from = {}
        self.cost_so_far = {}
        heap = [(0, start)]
        self.cost_so_far[start] = 0
        current = start
        
        while heap:
            if current == end:
                break
            current = heapq.heappop(heap)[1]
            for next_node in self.graph.neighbors(current):
                new_cost = self.cost_so_far[current] + self.heuristic(current, next_node) + self.heuristic(next_node, end)
                if next_node not in self.cost_so_far or new_cost < self.cost_so_far[next_node]: #self.grid[next_node.y, next_node.x] != self.States.OBS.value
                    self.cost_so_far[next_node] = new_cost
                    heapq.heappush(heap, (new_cost, next_node))
                    self.came_from[next_node] = current
        
        return self.reconstruct_path(start, end)
    
class Environment:
    def __init__(self, obstacle_density):
        # spawn grid
        self.grid = np.zeros((HEIGHT, WIDTH))
        self.obstacle_density = obstacle_density
        
        # spawn drone
        indices = np.argwhere(self.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.starting_pos = Point(indices[0,1], indices[0,0])
        self.pos = self.starting_pos
        self.prev_pos = self.starting_pos
        self.grid[self.starting_pos.y, self.starting_pos.x] = States.ROBOT.value

        # initialise exploration grid
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[y,x] != States.UNEXP.value: self.exploration_grid[y, x] = True

    def reset(self, weight, i):
        # spawn grid
        self.grid = np.zeros((HEIGHT, WIDTH))

        self.weight = weight
        # obstacles
        if i == 0:
            self.starting_grid = self.grid.copy()
            
                # Calculate the number of elements to be filled with 1's
            total_elements = int(HEIGHT) * int(WIDTH)
            num_ones_to_place = int(self.obstacle_density * total_elements)

                # Generate random indices to place 1's
            possible_indexes = np.argwhere(np.array(self.grid) == States.UNEXP.value)
            np.random.shuffle(possible_indexes)
            indexes = possible_indexes[:num_ones_to_place]

                # Set the elements at the random indices to 1
            self.starting_grid[indexes[:, 0], indexes[:, 1]] = States.OBS.value

            ES = Enclosed_space_check(HEIGHT, WIDTH, self.starting_grid, States)
            self.starting_grid = ES.enclosed_space_handler()
        self.grid = self.starting_grid.copy()
        
        # spawn drone
        indices = np.argwhere(self.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.starting_pos = Point(indices[0,1], indices[0,0])
        self.pos = self.starting_pos
        self.prev_pos = self.starting_pos
        self.grid[self.starting_pos.y, self.starting_pos.x] = States.ROBOT.value

        # initialise exploration grid
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[y,x] != States.UNEXP.value: self.exploration_grid[y, x] = True
        
    def move(self, new_pos):        
        # move drone to new position
        self.prev_pos = self.pos
        self.pos = Point(new_pos.x,new_pos.y)
        
        # update grids
        self.grid[self.prev_pos.y, self.prev_pos.x] = States.EXP.value
        self.grid[self.pos.y, self.pos.x] = States.ROBOT.value
        self.exploration_grid[self.pos.y, self.pos.x] = True

    def get_distance(self, end, start):
        return abs(start.x - end.x) + abs(start.y - end.y)

    def get_closest_unexplored(self):
        distances = {}
        temp_exploration_grid = self.exploration_grid.copy()
        temp_exploration_grid[self.pos.y, self.pos.x] = True
        
        # gets the distance to all unvisited blocks
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if temp_exploration_grid[y,x] == False:
                    distance = self.get_distance(Point(x,y), self.pos)

                    distances[Point(x,y)] = distance
        
        # checks if cell reachable
        if not distances:
            return None
        else:
            return min(distances, key=distances.get) # returns position
    
    def cost_function(self):
        # distances = {}
        temp_exploration_grid = self.exploration_grid.copy()
        temp_exploration_grid[self.pos.y, self.pos.x] = True
        
        # gets the distance to all unvisited blocks
        # for y in range(self.grid.shape[0]):
        #     for x in range(self.grid.shape[1]):
        #         if not temp_exploration_grid[y,x]:
        #             distance = self.get_distance(Point(x,y), self.pos)
        #             distances[Point(x,y)] = distance

        # if not distances:
        #     return None

        # divid into different clusters of unexplored regions
        cnter = 0
        labels = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if temp_exploration_grid[y,x]:
                    continue
                if cnter == 0:
                    labels.append([(x,y)])
                    cnter += 1
                
                # checks if (x,y) is in region
                # if not then add to new region
                added = False
                for i in range(len(labels)):
                    if (x,y) in labels[i]:
                        added = True
                        index = i
                if not added:
                    labels.append([(x,y)])
                    index = len(labels)-1

                # checks if right is connected
                if x != WIDTH-1:
                    if temp_exploration_grid[y,x] == temp_exploration_grid[y,x+1]:
                        # checks if (x+1,y) is in a label list already
                        combined = False
                        for i, l in enumerate(labels):
                            if (x+1,y) in labels[i] and i != index:
                                combine_lists = labels[i] + labels[index]
                                labels[min(i, index)] = combine_lists
                                del labels[max(i, index)]
                                combined = True
                        if not combined: 
                            labels[index].append((x+1,y))

                # checks if bottom is connected
                if y != HEIGHT-1:
                    if temp_exploration_grid[y,x] == temp_exploration_grid[y+1,x]:
                        # checks if (x+1,y) is in a label list already
                        combined = False
                        for i, l in enumerate(labels):
                            if (x,y+1) in labels[i] and i != index:
                                combine_lists = labels[i] + labels[index]
                                labels[min(i, index)] = combine_lists
                                del labels[max(i, index)]
                                combined = True
                        if not combined:
                            labels[index].append((x,y+1))
        
        # find shorest distance to clusters
        clusters = {}
        if len(labels) == 1:
            # Access the nested dictionary for drone position
            point_paths = distances[env.pos].copy()
            
            # remove points that have been explored
            pop_points = []
            for point in point_paths:
                if env.exploration_grid[point.y, point.x]:
                    pop_points.append(point)
            for point in pop_points:
                point_paths.pop(point)

            # # Find the shortest path in the nested dictionary
            # min_distance = min(point_paths.values(), key=len)

            # Find the key with the shortest path in the nested dictionary
            min_point = min(point_paths, key=lambda k: len(point_paths[k]))

            if (min_point.x, min_point.y) in labels[0]:
                clusters[min_point] = len(labels[0])
        else:
            for cluster in labels:
                lowest_value = 1000
                for key in cluster:
                    if Point(key[0],key[1]) in distances[env.pos] and len(distances[env.pos][Point(key[0],key[1])]) < lowest_value:
                        lowest_key = key
                        lowest_value = len(distances[env.pos][Point(key[0],key[1])])
                clusters[Point(lowest_key[0], lowest_key[1])] = len(cluster)

        costs = {}
        for point in clusters:
            if len(distances[point]) > 5:
                costs[point] = (WIDTH*HEIGHT)/len(distances[env.pos][point])
            else:
                # costs[point] = 1/len(distances[env.pos][point])
                costs[point] = clusters[point] / len(distances[point])
                # costs[point] = (clusters[point]) + self.weight/len(distances[env.pos][point])
                # costs[point] = (1 - len(distances[point])/(WIDTH + HEIGHT - 1)) + (clusters[point] / (WIDTH*HEIGHT))

        return max(costs, key=costs.get)
    
    def print_graph(self, steps, path, actions, starting_pos, obstacles, dir_path, index=0):
        """
        Prints the grid environment
        """

        plt.rc('font', size=20)
        plt.rc('axes', titlesize=10)

        # Prints graph
        fig,ax = plt.subplots(figsize=(env.grid.shape[1], env.grid.shape[0]))

        ax.set_aspect("equal")
        ax.set_xlim(0.5, WIDTH + 0.5)
        ax.set_ylim(0.5, HEIGHT + 0.5)
        # Set tick positions to be centered between grid lines
        ax.set_xticks(np.arange(WIDTH) + 0.5)
        ax.set_yticks(np.arange(HEIGHT) + 0.5)

        # Set tick labels to be the x or y coordinate of the grid cell
        ax.set_xticklabels(np.arange(WIDTH))
        ax.set_yticklabels(np.arange(HEIGHT))

        # Adjust tick label position and font size
        ax.tick_params(axis='both', labelsize=10, pad=2, width=0.5, length=2)
        ax.grid(True, color='black', linewidth=1)

        for i in range(env.grid.shape[0]): # y
            for j in range(env.grid.shape[1]): # x
                ax.fill([j+0.5, j + 1.5, j + 1.5, j+0.5], [i+0.5, i+0.5, i + 1.5, i + 1.5], facecolor="white", alpha=0.5)
                if obstacles[i,j] == True: ax.fill([j+0.5, j + 1.5, j + 1.5, j+0.5], [i+0.5, i+0.5, i + 1.5, i + 1.5], facecolor="k", alpha=0.5)
                elif Point(j,i) == starting_pos:
                    ax.fill([j + 0.5, j + 1.5, j + 1.5, j + 0.5],\
                            [i + 0.5, i + 0.5, i + 1.5, i + 1.5], \
                                facecolor="red", alpha=0.5)
                    
        # fill explored cells green
        for i, pos in enumerate(path):
            x = pos.x
            y = pos.y
            if i == len(path)-1:
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                    [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                    facecolor="yellow", 
                    alpha=0.5)
            else:
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                        [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                        facecolor="green", 
                        alpha=0.5)
            
        ax.fill([starting_pos.x + 0.5, starting_pos.x + 1.5, starting_pos.x + 1.5, starting_pos.x + 0.5],\
                            [starting_pos.y + 0.5, starting_pos.y + 0.5, starting_pos.y + 1.5, starting_pos.y + 1.5], \
                                facecolor="red", alpha=0.5)
            
        # adds all indices of actions on cell
        indices = {}
        for x in range(env.grid.shape[0]): # y
            for y in range(env.grid.shape[1]): # x
                for i, pos in enumerate(path):
                    if pos == Point(x,y): # if cell in path then add index to dict
                        if pos in indices: # checks if dict already has key named "pos"
                            indices[pos].append(i)
                        else:
                            indices[pos] = [i]
        
        clabel = ""
        for x in range(env.grid.shape[0]): # y
            for y in range(env.grid.shape[1]): # x
                if Point(x,y) in path:
                    for i in indices[Point(x,y)]:
                        if i == len(actions): break
                        if actions[i] == "right": 
                            clabel += "\u2192"
                            breakpoint
                        elif actions[i] == "left": 
                            clabel += "\u2190"
                            breakpoint
                        elif actions[i] == "up": 
                            clabel += "\u2193"
                            breakpoint
                        elif actions[i] == "down": 
                            clabel += "\u2191"
                            breakpoint

                temp_label = ""
                if len(clabel) > 3:
                    for j, c in enumerate(clabel):
                        temp_label += clabel[j:j+8] + "\n"
                    clabel = temp_label
                
                ax.text(x+1, y+1, clabel, ha="center", va="center", color="black", fontsize=8)
                clabel = ""
        
        plt_title = "A-star algorithm: Steps: %s" %(str(steps))
        plt.title(plt_title)

        
        file_name = "traj%d.png"%(index)
        plt.savefig(os.path.join(dir_path, file_name))
        plt.close()

save_trajectory = True
env = Environment(0)
weight = 1
env.reset(weight,0)
astar = Astar(HEIGHT, WIDTH, env.grid)

print("Preprocessing...")

distances = {}
for dx in range(WIDTH):
    for dy in range(HEIGHT):
        if env.grid[dy,dx] == States.OBS.value: continue
        paths = {}
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if env.grid[y,x] != States.OBS.value and (x,y) != (dx,dy):
                    path = astar.a_star(Point(dx, dy), Point(x,y))
                    del path[0]
                    paths[Point(x,y)] = path
                    distances[Point(dx,dy)] = paths

################################################################
# weights test
# weight = np.arange(1,20)
# indices = []
# for j in range(1000):
#     if j % 10 == 0 and j != 0: print(j, weight[max(set(indices), key = indices.count)])
#     env = Environment(0)
#     all_steps = []
#     for i,w in enumerate(weight):
#         env.reset(w,i)
#         astar = Astar(HEIGHT, WIDTH, env.grid)
#         # print(env.grid)
#         obstacles = env.exploration_grid.copy()
#         steps = 0
#         actions = []
#         trajectory = [env.starting_pos]
#         while not env.exploration_grid.all():
#             closest = env.cost_function()
#             path = astar.a_star(env.pos, closest)
#             del path[0]
#             while len(path) > 0:
#                 trajectory.append(path[0])
#                 steps += 1
#                 env.move(path[0])
#                 del path[0]
#                 if env.prev_pos.x < env.pos.x: actions.append("right")
#                 if env.prev_pos.x > env.pos.x: actions.append("left")
#                 if env.prev_pos.y > env.pos.y: actions.append("up")
#                 if env.prev_pos.y < env.pos.y: actions.append("down")
#                 # print(actions[steps-1], env.pos)

#         # print(w, " - ", steps)
#         all_steps.append(steps)

#     indices.append(all_steps.index(min(all_steps)))
# print(weight[max(set(indices), key = indices.count)])


################################################################
# steps test
if save_trajectory:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'Astar')      

    date_and_time = datetime.now()
    dir_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(dir_path): os.makedirs(dir_path)
all_steps = []
for j in range(1000):
    if j % 100 == 0 and j != 0: print(j)
    env = Environment(0)
    
    env.reset(weight,0)
    astar = Astar(HEIGHT, WIDTH, env.grid)
    # print(env.grid)
    obstacles = env.exploration_grid.copy()
    steps = 0
    actions = []
    trajectory = [env.starting_pos]
    while not env.exploration_grid.all():
        if steps > 60:
            breakpoint
        closest = env.cost_function()
        path = astar.a_star(env.pos, closest)
        del path[0]
        while len(path) > 0:
            trajectory.append(path[0])
            steps += 1
            env.move(path[0])
            del path[0]
            if env.prev_pos.x < env.pos.x: actions.append("right")
            if env.prev_pos.x > env.pos.x: actions.append("left")
            if env.prev_pos.y > env.pos.y: actions.append("up")
            if env.prev_pos.y < env.pos.y: actions.append("down")
            # print(actions[steps-1], env.pos)

        # print(w, " - ", steps)
    all_steps.append(steps)
    if save_trajectory and steps > 60:
        env.print_graph(steps, trajectory, actions, env.starting_pos, obstacles, dir_path, j)

print(np.mean(np.array(all_steps)))

    

###############################################################
# run simulations
# env = Environment(0)
# weight = 1
# env.reset(weight,0)
# astar = Astar(HEIGHT, WIDTH, env.grid)

# distances = {}
# for dx in range(WIDTH):
#     for dy in range(HEIGHT):
#         if env.grid[dy,dx] == States.OBS.value: continue
#         paths = {}
#         for x in range(WIDTH):
#             for y in range(HEIGHT):
#                 if env.grid[y,x] != States.OBS.value and (x,y) != (dx,dy):
#                     path = astar.a_star(Point(dx, dy), Point(x,y))
#                     del path[0]
#                     paths[Point(x,y)] = path
#                     distances[Point(dx,dy)] = paths

# print("Preprocessing done...")
# # print(env.grid)
# obstacles = env.exploration_grid.copy()
# steps = 0
# actions = []
# trajectory = [env.starting_pos]
# while not env.exploration_grid.all():
#     closest = env.cost_function()
#     path = astar.a_star(env.pos, closest)
#     del path[0]
#     while len(path) > 0:
#         trajectory.append(path[0])
#         steps += 1
#         env.move(path[0])
#         del path[0]
#         if env.prev_pos.x < env.pos.x: actions.append("right")
#         if env.prev_pos.x > env.pos.x: actions.append("left")
#         if env.prev_pos.y > env.pos.y: actions.append("up")
#         if env.prev_pos.y < env.pos.y: actions.append("down")
#         # print(actions[steps-1], env.pos)

if save_trajectory:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'Astar')      

    date_and_time = datetime.now()
    dir_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(dir_path): os.makedirs(dir_path)

    env.print_graph(steps, trajectory, actions, env.starting_pos, obstacles, dir_path)
