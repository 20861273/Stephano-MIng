import heapq
from collections import namedtuple
import collections
from enum import Enum
import numpy as np
from enclosed_space_checker import Enclosed_space_check
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
import json
from itertools import product
import pickle
from collections import deque
import math

Point = namedtuple('Point', 'x, y')
HEIGHT = 20
WIDTH = 20

# Chosen values
height = 100 # m

# Sony RX1R II
focal_length = 35.0 # mm
V = 2
H = 3
aspect_ratio = V/H
sensor_w = 35.9 # mm
sensor_h = 24.0 # mm
num_pixels = 42.4 * pow(10, 6)
pixel_w = num_pixels / aspect_ratio
pixel_h = num_pixels / pixel_w

################################ CALCULATING FOV FROM HIEGHT ################################################################################################
GSD_W = (height * sensor_w) / (focal_length * pixel_w) # m
GSD_H = (height * sensor_h) / (focal_length * pixel_h) # m

FOV_W = GSD_W * pixel_w
FOV_H = GSD_H * pixel_h

cell_dimensions = min(FOV_W, FOV_H)

# velocity
VEL = 16

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)  # Convert numpy int64 to Python int
        return json.JSONEncoder.default(self, obj)

def convert_data_to_json_format(data):
    # Convert namedtuple keys to strings for JSON serialization
    data_json = {str(str(key.x)+","+str(key.y)): item for key, item in data.items()}
    for key1, item1 in data_json.items():
        data_json[key1] = {str(str(key2.x)+","+str(key2.y)): item2 for key2, item2 in item1.items()}
    return data_json

def convert_json_data(data):
    data_json = {Point(int(key[0]),int(key[2])): item for key, item in data.items()}
    for key1, item1 in data_json.items():
        data_json[key1] = {Point(int(key2[0]),int(key2[2])): item2 for key2, item2 in item1.items()}
    return data_json

def write_json(lst, file_name):
    with open(file_name, "w") as f:
        json.dump(lst, f)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
    
def convert_grid_to_obstacle_positions(grids):
    obstacle_positions = [[] for _ in range(len(grids))]
    for i, grid in enumerate(grids):
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if grid[y][x] == States.OBS.value:
                    obstacle_positions[i].append(int(WIDTH/2*y+x))
    return obstacle_positions

def convert_coordinates_to_cells(starting_positions):
    positions = [[] for _ in range(len(starting_positions))]
    for i, starting_position in enumerate(starting_positions):
        for r in range(nr):
            x = starting_position[r].x
            y = starting_position[r].y
            positions[i].append(int(WIDTH/2*y+x))
    return positions

def convert_target_coordinates_to_cells(goal_positions):
    goals = [[] for _ in range(len(goal_positions))]
    for i, goal_position in enumerate(goal_positions):
        x = goal_position.x
        y = goal_position.y
        goals[i].append(int(WIDTH/2*y+x))
    return goals


class Grid(object):
    def __init__(self, height, width, grid, direction): ########################################### change
        self.grid = grid
        self.width = width
        self.height = height

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, id, current_id, path_step):
        (x, y) = id
        temp_grid = env.grid.copy()
        # if not preprocessing:
        #     if path_step == 0:
        #         for ri in range(nr):
        #             if r > ri:
        #                 # on location collision
        #                 # make next position of drone an obstacle 
        #                 temp_grid[env.pos[ri].y, env.pos[ri].x] = States.OBS.value

        #                 # cross location collision
        #                 # if current location of drone is equal to next position id
        #                 # and previous location of drone is equal to current position id
        #                 # make current position id obstacle
        #                 if env.prev_pos[ri] ==  Point(x,y)\
        #                     and env.pos[ri] == Point(current_id[0],current_id[1]):
        #                     temp_grid[y, x] = States.OBS.value

        return temp_grid[y,x] != States.OBS.value

    def neighbors(self, id, path_step, direction):
        (x, y) = id
        
        if preprocessing or not fixed_wing:
            # (right, up, left, down)
            results = [Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        else:
            if direction == "right":
                results = [Point(x+1, y), Point(x, y-1), Point(x, y+1)]
            elif direction == "left":
                results = [Point(x, y-1), Point(x-1, y), Point(x, y+1)]
            elif direction == "up":
                results = [Point(x+1, y), Point(x, y-1), Point(x-1, y)]
            elif direction == "down":
                results = [Point(x+1, y), Point(x-1, y), Point(x, y+1)]
        
        # This is done to prioritise straight paths
        #if (x + y) % 2 == 0: results.reverse()
        results = list(filter(self.in_bounds, results))
        results = list(filter(lambda k: self.is_obstacle(k, id, path_step), results))
        if len(results) == 0:
            breakpoint
        # if occupied_cells != None:
        #     results = list(filter(lambda k: self.is_collision(k, cr, x, y, path_step), results))
        # if len(results) == 0:
        #     breakpoint
        return results

class Astar:
    def __init__(self, grid):
        self.came_from = {}
        self.cost_so_far = {}
        self.grid = grid

    # Heuristic function to estimate the cost from a current block to a end block
    def heuristic(self, current, end):
        return abs(current.x - end.x) + abs(current.y - end.y)
    
    # Path reconstruction
    def reconstruct_path(self, start, end, id=None): #: dict[Location, Location], : Location, : Location
        current = end
        path = [current]
        if maneuvering: reconstructed_maneuvers = [maneuvers[r][steps-1]]
        while (id,current) != (0,start):
            if (id,current) not in self.came_from:
                print((id,current) ,self.came_from)
                print(self.grid, self.start, end, "this one")
                
            if maneuvering:
                maneuver = self.came_from_maneuvers[(id, current)][2]
            new_id, new_current = self.came_from[(id, current)]
            current = new_current
            id = new_id
            path.append(current)
            if maneuvering: reconstructed_maneuvers.append(maneuver)
        path.reverse()
        if maneuvering:
            reconstructed_maneuvers.reverse()
            return path, reconstructed_maneuvers
        else:
            return path

    # A* algorithm
    def a_star(self, start, end, grid, dynamic_obstacles, direction=None):
        self.grid = grid
        self.dynamic_obstacles = dynamic_obstacles
        self.graph = Grid(HEIGHT, WIDTH, grid, direction)
        self.start = start
        self.came_from = {}
        self.cost_so_far = {}
        self.heap = [(0, 0, start, direction)]
        if maneuvering:
            self.came_from_maneuvers = {(0,start):(-1, env.prev_pos[r], maneuvers[r][steps-1])}
            # self.cost_so_far[start] = 0
        self.cost_so_far[(0,start)] = 0
        current = start
        found = False
        id = 0
        
        while len(self.heap) > 0:
            _, id, current, direction = heapq.heappop(self.heap)
            path_step = id
            if current == end:
                found = True
                break
            
            if id > HEIGHT*WIDTH:
                breakpoint
                print("Maybe stuck...")
            
            self.neighbors = self.graph.neighbors(current, path_step, direction)
            for next_node in self.neighbors:
                maneuver = False
                # on location collision
                # if a drone has planned thus far
                # AND a drone is on next_node location
                # AND current is not a maneuver action
                if steps+path_step in self.dynamic_obstacles and next_node in self.dynamic_obstacles[steps+path_step]:
                    if maneuvering:
                        # if steps+path_step-2 in self.dynamic_obstacles and current not in self.dynamic_obstacles[steps+path_step-2]:
                        if current in self.came_from_maneuvers and self.came_from_maneuvers[current][1]:
                            continue
                        maneuver = True
                    else:
                        continue
                # cross location collision
                # current is in dynamic obstacles on next step
                if steps+path_step in self.dynamic_obstacles and current in self.dynamic_obstacles[steps+path_step]:
                    # next node is in dynamic obstacles on previous step
                    if steps+path_step-1 in self.dynamic_obstacles and next_node in self.dynamic_obstacles[steps+path_step-1]:
                        if maneuvering:
                            maneuver = True
                        else:
                            continue

                if maneuvering and maneuver:
                    new_cost = self.cost_so_far[(id,current)] + self.heuristic(current, next_node) + 3
                else:
                    new_cost = self.cost_so_far[(id,current)] + self.heuristic(current, next_node)
            
                if (id+1,next_node) not in self.cost_so_far:
                    self.cost_so_far[(id+1,next_node)] = new_cost
                    if fixed_wing:
                        if current.x < next_node.x: # right
                            direction = "right"
                        elif current.x > next_node.x: # left
                            direction = "left"
                        elif current.y < next_node.y: # down
                            direction = "down"
                        elif current.y > next_node.y: # up
                            direction = "up"
                    priority = new_cost + self.heuristic(next_node, end)
                    heapq.heappush(self.heap, (priority, id+1, next_node, direction))
                    self.came_from[(id+1,next_node)] = (id, current)
                    if maneuvering: self.came_from_maneuvers[(id+1,next_node)] = (id, current, maneuver)
        
        if current == end:
            found = True
        # if current != end:
        #     print("bugga%d"%(i))
        if found:
            if maneuvering:
                return self.reconstruct_path(start, end, id)
            else:
                self.which = 1
                return self.reconstruct_path(start, end, id), []
        else:
            return None, None
    
class Environment:
    def __init__(self, nr, obstacles, set_obstacles, obstacle_density, save_obstacles, save_dir, load_obstacles, load_dir):
        # initalise variables
        self.nr = nr
        self.obstacles = obstacles
        self.set_obstacles = set_obstacles
        self.obstacle_density = obstacle_density
        self.save_obstacles = save_obstacles
        self.save_dir = save_dir
        self.load_obstacles = load_obstacles
        self.load_dir = load_dir
        
        # spawn grid
        self.starting_grid = np.zeros((HEIGHT, WIDTH))
        self.grid = self.starting_grid.copy()
        self.grids = []
        self.goals = [[] for _ in range(test_iterations)]
        
        # initialise drone
        self.starting_pos = [None]*nr
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()
        self.starting_positions = [[] for _ in range(test_iterations)]

        # initialise exploration grid
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.starting_grid[y,x] != States.UNEXP.value: self.exploration_grid[y, x] = True
        
        if self.load_obstacles:
            file_name = "positions.pkl"
            file_path = os.path.join(self.load_dir, file_name)
            with open(file_path, 'rb') as file:
                # Deserialize and read the sublists using pickle.load()
                self.starting_positions = pickle.load(file)
            
            file_name = "grid.json"
            file_name = os.path.join(self.load_dir, file_name)
            self.grids = np.array(read_json(file_name))

            file_name = "targets.pkl"
            file_path = os.path.join(self.load_dir, file_name)
            with open(file_path, 'rb') as file:
                # Deserialize and read the sublists using pickle.load()
                self.goals = pickle.load(file)

        breakpoint

    def reset(self, weight, i, goal_spawning):
        # spawn grid
        global WIDTH
        global HEIGHT
        self.grid = np.zeros((HEIGHT, WIDTH))

        self.weight = weight
        # obstacles
        if self.obstacles:
            if self.load_obstacles:
                self.starting_grid = np.array(self.grids[i])
                self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))
                self.grid = self.starting_grid.copy()
            elif preprocessing or not self.set_obstacles:
                self.starting_grid = np.zeros((int(HEIGHT/2), int(WIDTH/2)), dtype=np.int8)
                
                    # Calculate the number of elements to be filled with 1's
                total_elements = int(HEIGHT/2) * int(WIDTH/2)
                num_ones_to_place = math.ceil(self.obstacle_density * total_elements)

                    # Generate random indices to place 1's
                possible_indexes = np.argwhere(np.array(self.starting_grid) == States.UNEXP.value)
                np.random.shuffle(possible_indexes)
                indexes = possible_indexes[:num_ones_to_place]

                    # Set the elements at the random indices to 1
                self.starting_grid[indexes[:, 0], indexes[:, 1]] = States.OBS.value

                # self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))

                WIDTH = int(WIDTH/2)*2
                HEIGHT = int(HEIGHT/2)*2

                self.ES = Enclosed_space_check(int(HEIGHT/2), int(WIDTH/2), self.starting_grid, States)
                self.starting_grid = self.ES.enclosed_space_handler()

                self.grids.append(self.starting_grid.tolist())

                self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))
                self.grid = self.starting_grid.copy()

        self.grid = self.starting_grid.copy()
        
        # spawn drone
        if self.load_obstacles and self.starting_positions[0] != []:
            self.starting_pos = [Point(self.starting_positions[i][ri].x*2, self.starting_positions[i][ri].y*2) for ri in range(nr)]
            for ri in range(self.nr):
                self.pos[ri] = self.starting_pos[ri]
                self.prev_pos[ri] = self.starting_pos[ri]
                self.grid[self.starting_pos[ri].y, self.starting_pos[ri].x] = States.ROBOT.value
        else:
            if self.set_obstacles:
                indices = np.argwhere(np.array(self.grids[0]) == States.UNEXP.value)
            else:
                indices = np.argwhere(np.array(self.grids[i]) == States.UNEXP.value)
            np.random.shuffle(indices)
            save_starting_pos = [None]*nr
            for ri in range(self.nr):
                save_starting_pos[ri] = Point(indices[0,1], indices[0,0])
                self.starting_pos[ri] = Point(indices[0,1]*2, indices[0,0]*2)
                self.pos[ri] = self.starting_pos[ri]
                self.prev_pos[ri] = self.starting_pos[ri]
                self.grid[self.starting_pos[ri].y, self.starting_pos[ri].x] = States.ROBOT.value
                indices_list = indices.tolist()
                del indices_list[0]
                indices = np.array(indices_list)

        # set directions
        self.direction = [None]*self.nr
        for ri in range(self.nr):
            if env.pos[ri].x <= WIDTH/2 and env.pos[ri].y <= HEIGHT/2: # top left
                self.direction[ri] = "right"
            elif env.pos[ri].x <= WIDTH/2 and env.pos[ri].y > HEIGHT/2: # bottom left
                self.direction[ri] = "right"
            elif env.pos[ri].x > WIDTH/2 and env.pos[ri].y <= HEIGHT/2: # top right
                self.direction[ri] = "left"
            elif env.pos[ri].x > WIDTH/2 and env.pos[ri].y > HEIGHT/2: # bottom right
                self.direction[ri] = "left"
        
        # spawn goal
        if goal_spawning:
            if self.load_obstacles:
                self.goal = Point(self.goals[i].x*2, self.goals[i].y*2)
            else:
                if self.set_obstacles:
                    indices = np.argwhere(np.array(self.grids[0]) == States.UNEXP.value)
                else:
                    indices = np.argwhere(np.array(self.grids[i]) == States.UNEXP.value)
                np.random.shuffle(indices)
                save_goal_pos = Point(indices[0,1], indices[0,0])
                self.goal = Point(indices[0,1]*2, indices[0,0]*2)

        if self.save_obstacles:
            self.starting_positions[i] = save_starting_pos.copy()
            self.goals[i] = save_goal_pos

        # initialise exploration grid
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[y,x] != States.UNEXP.value: self.exploration_grid[y, x] = True
        for r in range(self.nr): self.exploration_grid[self.pos[r].y, self.pos[r].x] = True

        # set target cluster
        self.target_cluster = [[None]]*self.nr
        
    def move(self, r, new_pos):        
        # move drone to new position
        self.prev_pos[r] = self.pos[r]
        self.pos[r] = Point(new_pos.x,new_pos.y)
        
        # update grids
        self.grid[self.prev_pos[r].y, self.prev_pos[r].x] = States.EXP.value
        self.grid[self.pos[r].y, self.pos[r].x] = States.ROBOT.value
        self.exploration_grid[self.prev_pos[r].y, self.prev_pos[r].x] = True
        self.exploration_grid[self.pos[r].y, self.pos[r].x] = True

        if self.prev_pos[r].x < self.pos[r].x: # right
            self.direction[r] = "right"
        elif self.prev_pos[r].x > self.pos[r].x: # left
            self.direction[r] = "left"
        elif self.prev_pos[r].y < self.pos[r].y: # down
            self.direction[r] = "down"
        elif self.prev_pos[r].y > self.pos[r].y: # up
            self.direction[r] = "up"

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
        
    def get_min_targets(self, costs):
        # get min value of each dorne
        min_targets_value = [None]*self.nr
        for ri in range(self.nr):
            if not costs[ri]: min_targets_value[ri] = []
            else: min_targets_value[ri] = min(costs[ri].values())

        # min_targets_value = [None]*self.nr
        # for ri in range(self.nr):
        #     min_targets_value[ri] = np.min(np.ma.masked_equal(dists[self.pos[ri].y*HEIGHT+self.pos[ri].x], 0, copy=False))

        # get all positions with min value
        min_targets = [[] for i in range(self.nr)]
        for ri in range(self.nr):
            min_targets[ri] = [key for key, value in costs[ri].items() if costs[ri] if value == min_targets_value[ri] ]

        # min_targets = [[] for i in range(self.nr)]
        # for ri in range(self.nr):
        #     cells = np.argwhere(dists[self.pos[ri].y*HEIGHT+self.pos[ri].x] == min_targets_value[ri])
        #     min_targets[ri] = [Point(c // WIDTH, c % WIDTH) for c in cells[0]]
        
        return min_targets
        
    def scheduler(self, ongoing_frontiers):
        # set current positions to explored
        temp_exploration_grid = self.exploration_grid.copy()
        for ri in range(self.nr): 
            temp_exploration_grid[self.pos[ri].y, self.pos[ri].x] = True

        # if no more unexplored cells return to home
        if np.count_nonzero(temp_exploration_grid) == HEIGHT*WIDTH:
            return [self.starting_pos[ri] for ri in range(self.nr)]

        # if on going frontier already searched
        for ri in range(nr):
            if ongoing_frontiers[ri] == None: continue
            if temp_exploration_grid[ongoing_frontiers[ri].y, ongoing_frontiers[ri].x]:
                ongoing_frontiers[ri] = None
        
        # gets the distance to all unvisited blocks
        if in_loop_dist:
            for ri in range(self.nr):
                if ongoing_frontiers[ri] != None: continue
                temp_dist = {}
                for y in range(self.grid.shape[0]):
                    for x in range(self.grid.shape[1]):
                        if not temp_exploration_grid[y,x]:
                            distance = self.get_distance(Point(x,y), self.pos[ri])
                            temp_dist[Point(x,y)] = distance
                            distances[self.pos[ri]] = temp_dist

        # get costs of unexplored cells
        costs = [{} for _ in range(self.nr)]
        for ri in range(self.nr):
            # if ongoing_frontiers[ri] != None: continue
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    if not temp_exploration_grid[y,x]:
                        # costs[ri][Point(x,y)] = distances[self.pos[ri]][Point(x,y)]
                        costs[ri][Point(x,y)] = distances[self.pos[ri].y*HEIGHT + self.pos[ri].x][y*HEIGHT + x]

        # set minimum targets
        min_targets = self.get_min_targets(costs)
        temp_costs = [{key: value for key, value in dictionary.items()} for dictionary in costs]
        for rj in range(self.nr-1):
            # delete best targets from temp cost list
            for ri in range(self.nr):
                for target in min_targets[ri]:
                    if target in temp_costs[ri]: del temp_costs[ri][target]

            # find next best targets
            next_min_targets = self.get_min_targets(temp_costs)
            for ri in range(self.nr):
                if ongoing_frontiers[ri] != None:
                    min_targets[ri] = [ongoing_frontiers[ri]]
                else:
                    min_targets[ri] += next_min_targets[ri]
        
        # # set minimum targets (if working would be faster, excludes costs for loops)
        # min_targets = self.get_min_targets(distances)
        # temp_distances = np.array(distances)
        # for rj in range(self.nr-1):
        #     # delete best targets from temp cost list
        #     for ri in range(self.nr):
        #         for target in min_targets[ri]:
        #             cells = np.argwhere(temp_distances[self.pos[ri].y*HEIGHT+self.pos[ri].x] == target)
        #             temp_distances[cells] = HEIGHT*WIDTH

        #     # find next best targets
        #     next_min_targets = self.get_min_targets(temp_distances)
        #     for ri in range(self.nr):
        #         if ongoing_frontiers[ri] != None:
        #             min_targets[ri] = [ongoing_frontiers[ri]]
        #         else:
        #             min_targets[ri] += next_min_targets[ri]

        # get all combinations of best targets
        combinations = list(product(*min_targets))

        # find invalid combinations
        delete_indices = []
        for i,combination in enumerate(combinations):
            if len(combination) != len(set(combination)):
                delete_indices.append(i)
            # check drone neighbours
            # if drone only has 1 valid neighbour mark node as invalid for other drones
        
        # delete invalid combinations
        modified_combinations = [combinations[i] for i in range(len(combinations)) if i not in delete_indices]
        combinations = modified_combinations.copy()

        # if no equal targets
        targets = [None]*self.nr
        if len(combinations) > 0:
            # find sum costs of combinations
            sum_costs = []
            for combination in combinations:
                sum_cost = 0
                for ri, target in enumerate(combination):
                    if target in costs[ri]:
                        sum_cost += costs[ri][target]
                    # sum_cost += distances[self.pos[ri].y*HEIGHT+self.pos[ri].x][target.y*HEIGHT+target.x]
                sum_costs.append(sum_cost)
            
            # set targets to best combination
            min_cost = min(sum_costs)
            best_combination = combinations[sum_costs.index(min_cost)]
            for ri in range(self.nr):
                targets[ri] = best_combination[ri]

            if not self.exploration_grid.all() and all([targets[ri] == self.starting_pos[ri] for ri in range(self.nr)]) and targets[0] == self.starting_pos[0]:
                breakpoint

            return targets
        
        # add starting positions to maximum targets with lower value
        for ri in range(self.nr):
            min_targets[ri].append(env.starting_pos[ri])

        # get all combinations of best targets
        combinations = list(product(*min_targets))

        # find invalid combinations
        delete_indices = []
        for i,combination in enumerate(combinations):
            if len(combination) != len(set(combination)):
                delete_indices.append(i)
        
        # delete invalid combinations
        modified_combinations = [combinations[i] for i in range(len(combinations)) if i not in delete_indices]
        combinations = modified_combinations.copy()
        
        # find sum costs of combinations
        sum_costs = []
        penalty = WIDTH+HEIGHT
        for combination in combinations:
            sum_cost = 0
            for ri, target in enumerate(combination):
                if target in costs[ri]:
                    sum_cost += costs[ri][target]
                # sum_cost += distances[self.pos[ri].y*HEIGHT+self.pos[ri].x][target.y*HEIGHT+target.x]
                if target == env.starting_pos[ri]:
                    sum_cost += penalty
            sum_costs.append(sum_cost)
        
        # set targets to best combination
        min_cost = min(sum_costs)
        best_combination = combinations[sum_costs.index(min_cost)]
        for ri in range(self.nr):
            targets[ri] = best_combination[ri]
            if self.pos[ri] != self.starting_pos[ri]:
                done[ri] = False
            if targets[ri] != self.starting_pos[ri]:
                done[ri] = False

        if not self.exploration_grid.all() and all([targets[ri] == self.starting_pos[ri] for ri in range(self.nr)]) and targets[0] == self.starting_pos[0]:
            breakpoint

        return targets
    
    def calculate_distances(self):
        num_cells = HEIGHT * WIDTH
        distances = np.full((num_cells, num_cells), HEIGHT * WIDTH)  # Initialize distances to HEIGHT*WIDTH (unreachable)

        # Define movements (up, down, left, right)
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for start_cell in range(num_cells):
            if self.starting_grid[start_cell // HEIGHT, start_cell % HEIGHT] == States.UNEXP.value:
                queue = deque([(start_cell, 0)])  # Initialize queue with the starting cell and distance 0
                visited = np.zeros(num_cells, dtype=bool)  # Mark all cells as not visited
                visited[start_cell] = True  # Mark the starting cell as visited

                while queue:
                    current_cell, distance = queue.popleft()
                    distances[start_cell, current_cell] = distance

                    # Explore neighbors
                    row, col = current_cell // HEIGHT, current_cell % HEIGHT
                    for move_row, move_col in moves:
                        new_row, new_col = row + move_row, col + move_col
                        neighbor_cell = new_row * HEIGHT + new_col

                        if 0 <= new_row < WIDTH and 0 <= new_col < HEIGHT \
                            and not visited[neighbor_cell] and self.starting_grid[new_row, new_col] == States.UNEXP.value:
                            queue.append((neighbor_cell, distance + 1))
                            visited[neighbor_cell] = True

        return distances
    
    def print_graph(self, r, steps, path, maneuvers, actions, starting_pos, obstacles, dir_path, cnt, summary=False, goal_pos=None, step=None):
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
                if obstacles[i,j] and Point(j,i) not in env.starting_pos:
                    ax.fill([j+0.5, j + 1.5, j + 1.5, j+0.5], [i+0.5, i+0.5, i + 1.5, i + 1.5], facecolor="k", alpha=0.5)
                elif Point(j,i) == goal_pos:
                    ax.fill([j + 0.5, j + 1.5, j + 1.5, j + 0.5],\
                            [i + 0.5, i + 0.5, i + 1.5, i + 1.5], \
                                facecolor="cyan", alpha=0.5)
                    
        # fill explored cells green
        for i, pos in enumerate(path):
            x = pos.x
            y = pos.y
            if i == 0:
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                    [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                    facecolor="green", 
                    alpha=0.5)
            elif i == len(path)-1 and goal_pos == None:
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                    [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                    facecolor="yellow", 
                    alpha=0.5)
            # elif i == len(path)-1 and goal_pos == None and maneuvers[i]:
            #     ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
            #         [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
            #         facecolor="red", 
            #         alpha=0.5)
            elif goal_pos != Point(x,y) and Point(x,y) != starting_pos:
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                        [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                        facecolor="blue", 
                        alpha=0.5)
            # elif goal_pos != Point(x,y) and maneuvers[i]:
            #     ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
            #             [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
            #             facecolor="red", 
            #             alpha=0.5)
            
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
                        if maneuvering:
                            if actions[i] == "right":
                                clabel += "%02d\u2192 "%(i)
                                breakpoint
                            elif actions[i] == "left" and not maneuvers[i]: 
                                clabel += "%02d\u2190 "%(i)
                                breakpoint
                            elif actions[i] == "up" and not maneuvers[i]: 
                                clabel += "%02d\u2193 "%(i)
                                breakpoint
                            elif actions[i] == "down" and not maneuvers[i]: 
                                clabel += "%02d\u2191 "%(i)
                                breakpoint
                            elif actions[i] == "right" and maneuvers[i]: 
                                clabel += "%02d\u21D2 "%(i)
                                breakpoint
                            elif actions[i] == "left" and maneuvers[i]: 
                                clabel += "%02d\u21D0 "%(i)
                                breakpoint
                            elif actions[i] == "up" and maneuvers[i]: 
                                clabel += "%02d\u21D3 "%(i)
                                breakpoint
                            elif actions[i] == "down" and maneuvers[i]: 
                                clabel += "%02d\u21D1 "%(i)
                                breakpoint
                        else:
                            if actions[i] == "right":
                                clabel += "%02d\u2192 "%(i)
                                breakpoint
                            elif actions[i] == "left": 
                                clabel += "%02d\u2190 "%(i)
                                breakpoint
                            elif actions[i] == "up": 
                                clabel += "%02d\u2193 "%(i)
                                breakpoint
                            elif actions[i] == "down": 
                                clabel += "%02d\u2191 "%(i)
                                breakpoint

                temp_label = ""
                if len(clabel) > 3:
                    for j in range(0, len(clabel), 8):
                        if len(clabel) > 8:
                            temp_label += clabel[j:j+8] + "\n"
                        else: temp_label += clabel[j::]
                    clabel = temp_label
                
                ax.text(x+1, y+1, clabel, ha="center", va="center", color="black", fontsize=8)
                clabel = ""
        
        if step != None:
            plt_title = "A-star algorithm drone %s: Steps: %s Collision step:%d" %(str(r) ,str(steps), step)
        else:
            plt_title = "A-star algorithm drone %s: Steps: %s" %(str(r) ,str(steps))
        plt.title(plt_title)
        if summary:
            file_name = "trajectory%d_drone%d.png"%(cnt, r)
        else:
            file_name = "trajectory%d_drone%d_step%d.png"%(cnt, r, steps)
        plt.savefig(os.path.join(dir_path, file_name))
        # plt.show()
        # plt.pause(0.0005)
        plt.close()

    def print_frame(self, steps, path, maneuvers, actions, starting_pos, obstacles, dir_path, cnt, summary=False, goal_pos=None, step=None):
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
                if obstacles[i,j] and Point(j,i) not in env.starting_pos:
                    ax.fill([j+0.5, j + 1.5, j + 1.5, j+0.5], [i+0.5, i+0.5, i + 1.5, i + 1.5], facecolor="k", alpha=0.5)
                elif Point(j,i) == goal_pos:
                    ax.fill([j + 0.5, j + 1.5, j + 1.5, j + 0.5],\
                            [i + 0.5, i + 0.5, i + 1.5, i + 1.5], \
                                facecolor="cyan", alpha=0.5)
                    
        # fill explored cells green
        for ri in range(self.nr):
            for i, pos in enumerate(path[ri]):
                x = pos.x
                y = pos.y
                if i == 0:
                    ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                        [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                        facecolor="green", 
                        alpha=0.5)
                elif i == len(path[ri])-1 and goal_pos == None:
                    ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                        [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                        facecolor="yellow", 
                        alpha=0.5)
                # elif i == len(path)-1 and goal_pos == None and maneuvers[i]:
                #     ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                #         [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                #         facecolor="red", 
                #         alpha=0.5)
                elif goal_pos != Point(x,y) and Point(x,y) != starting_pos:
                    ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                            [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                            facecolor="blue", 
                            alpha=0.5)
                # elif goal_pos != Point(x,y) and maneuvers[i]:
                #     ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                #             [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                #             facecolor="red", 
                #             alpha=0.5)
            
        # adds all indices of actions on cell
        indices = {}
        for x in range(env.grid.shape[0]): # y
            for y in range(env.grid.shape[1]): # x
                for ri in range(self.nr):
                    for i, pos in enumerate(path[ri]):
                        if pos == Point(x,y): # if cell in path then add index to dict
                            if pos in indices: # checks if dict already has key named "pos"
                                indices[(ri, pos)].append(i)
                            else:
                                indices[(ri, pos)] = [i]
        
        clabel = ""
        for x in range(env.grid.shape[0]): # y
            for y in range(env.grid.shape[1]): # x
                for ri in range(self.nr):
                    if Point(x,y) in path[ri]:
                        for i in indices[(ri, Point(x,y))]:
                            if i == len(actions[ri]): break
                            if maneuvering:
                                if actions[ri][i] == "right" and not maneuvers[ri][i]:
                                    clabel += "%02d\u2192 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "left" and not maneuvers[ri][i]: 
                                    clabel += "%02d\u2190 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "up" and not maneuvers[ri][i]: 
                                    clabel += "%02d\u2193 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "down" and not maneuvers[ri][i]: 
                                    clabel += "%02d\u2191 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "right" and maneuvers[ri][i]: 
                                    clabel += "%02d\u21D2 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "left" and maneuvers[ri][i]: 
                                    clabel += "%02d\u21D0 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "up" and maneuvers[ri][i]: 
                                    clabel += "%02d\u21D3 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "down" and maneuvers[ri][i]: 
                                    clabel += "%02d\u21D1 "%(i)
                                    breakpoint
                            else:
                                if actions[ri][i] == "right":
                                    clabel += "%02d\u2192 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "left": 
                                    clabel += "%02d\u2190 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "up": 
                                    clabel += "%02d\u2193 "%(i)
                                    breakpoint
                                elif actions[ri][i] == "down": 
                                    clabel += "%02d\u2191 "%(i)
                                    breakpoint

                temp_label = ""
                if len(clabel) > 3:
                    for j in range(0, len(clabel), 8):
                        if len(clabel) > 8:
                            temp_label += clabel[j:j+8] + "\n"
                        else: temp_label += clabel[j::]
                    clabel = temp_label
                
                ax.text(x+1, y+1, clabel, ha="center", va="center", color="black", fontsize=8)
                clabel = ""
        
        if step != None:
            plt_title = "A-star algorithm: Steps: %s Collision step:%d" %(str(steps), step)
        else:
            plt_title = "A-star algorithm: Steps: %s" %(str(steps))
        plt.title(plt_title)
        if summary:
            file_name = "trajectory%d.png"%(cnt)
        else:
            file_name = "trajectory%d_step%d.png"%(cnt, steps)
        plt.savefig(os.path.join(dir_path, file_name))
        plt.show()
        plt.pause(0.0005)
        # plt.close()

# weights test
# weight = np.arange(1,20)
# indices = []
# for j in range(1000):
#     if j % 10 == 0 and j != 0: print(j, weight[max(set(indices), key = indices.count)])
#     env = Environment(0.3)
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
#         if save_trajectory:
#             PATH = os.getcwd()
#             PATH = os.path.join(PATH, 'SAR')
#             PATH = os.path.join(PATH, 'Results')
#             PATH = os.path.join(PATH, 'Astar')      

#             date_and_time = datetime.now()
#             dir_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
#             if not os.path.exists(dir_path): os.makedirs(dir_path)

#             env.print_graph(trajectory, actions, env.starting_pos, obstacles, dir_path)

#     indices.append(all_steps.index(min(all_steps)))
# print(weight[max(set(indices), key = indices.count)])

# initialisations
PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
# set environment
# file_name = "distances.json"
# file_path = os.path.join(PATH, file_name)
# distances = read_json(file_path)
# distances = convert_json_data(distances)
save_trajectory = True
in_loop_trajectory = False
each_drone = False
in_loop_dist = False
preprocessing = True
maneuvering = False
fixed_wing = True
test_iterations = 10000
saved_iterations = 10
saves = []

# set directory path
if save_trajectory:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'Astar')      

    date_and_time = datetime.now()
    dir_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(dir_path): os.makedirs(dir_path)

# environment initialisations
goal_spawning = True
nr = 3
weight = 19
obstacles = True
obstacle_density = 0
set_obstacles = False
save_obstacles = False
save_dir = os.path.join(dir_path, 'Save')
if not os.path.exists(save_dir): os.makedirs(save_dir)
load_obstacles = True
load_dir = os.path.join(PATH, 'Load') 
if not os.path.exists(load_dir): os.makedirs(load_dir)
env = Environment(nr, obstacles, set_obstacles, obstacle_density, save_obstacles, save_dir, load_obstacles, load_dir)
env.reset(weight, 0, goal_spawning)

# calculate distances
if not in_loop_dist:
    # file_name = "distances%dx%d.bin"%(HEIGHT,WIDTH)
    file_name = "distances%dx%d.json"%(HEIGHT,WIDTH)
    if file_name in os.listdir(load_dir):
    #     file_path = os.path.join(load_dir, file_name)
    #     with open(file_path, 'rb') as file:
    #         distances = file.read()
        file_path = os.path.join(load_dir, file_name)
        distances = read_json(file_path)

        distances = convert_json_data(distances)
    else:
        distances = {}
        print("Preprocessing...")
        starting_time = time.time()
        # for dx in range(WIDTH):
        #     print(dx)
        #     for dy in range(HEIGHT):
        #         if env.grid[dy,dx] == States.OBS.value: continue
        #         temp_paths = {}
        #         for x in range(WIDTH):
        #             for y in range(HEIGHT):
        #                 if env.grid[y,x] != States.OBS.value and (x,y) != (dx,dy):
        #                     # A* distances
        #                     # astar = Astar(env.grid)
        #                     # temp_path = astar.a_star(Point(dx, dy), Point(x,y), env.grid)
        #                     # del temp_path[0]
        #                     # temp_paths[Point(x,y)] = temp_path
        #                     # distances[Point(dx,dy)] = temp_paths

        #                     # Mannhattan distances
        #                     distance = env.get_distance(Point(dx, dy), Point(x,y))
        #                     temp_paths[Point(x,y)] = distance
        #                     distances[Point(dx,dy)] = temp_paths

        # # file_name = "distances%dx%d.bin"%(HEIGHT,WIDTH)
        # # file_path = os.path.join(load_dir, file_name)
        # # with open(file_path, 'wb') as file:
        # #     file.write(distances)

        # # write to file for later use
        # # data_json = convert_data_to_json_format(distances)
        # # file_name = "distances%dx%d.json"%(HEIGHT,WIDTH)
        # # file_path = os.path.join(load_dir, file_name)
        # # write_json(data_json, file_path)

        distances = env.calculate_distances()

    end_time = time.time()
    print("Preprocessing time: %.2fs" %(end_time - starting_time))
    
preprocessing = False

# testing loop
testing_start_time = time.time()
steps_list = []
explorations_list = [[] for r in range(nr)]
maneuvers_list = []
planning_times = []
flight_times = []
flight_distances = []
schedule_times = []
path_times = []
frontier_list = [[] for _ in range(test_iterations)]
counter = 0
unsuccessful = 0
for i in range(test_iterations):
    successful_condition = True
    save = False
    planning_starting_time = time.time()
    if i % 500 == 0: print(i)
    if i != 0:
        env.reset(weight, i, goal_spawning)
        if not set_obstacles: distances = env.calculate_distances()
    obstacles = env.exploration_grid.copy()
    steps = 0
    actions = [[] for _ in range(nr)]
    trajectory = [[env.starting_pos[r]] for r in range(nr)]
    maneuvers = [[False] for r in range(nr)]
    current_path = [[] for _ in range(nr)]
    current_maneuvers = [[] for _ in range(nr)]
    done = [False]*nr
    which = [False]*nr
    occupied_cells = {}
    occupied_cells[steps] = []
    explorations = [0]*nr
    for r in range(nr):
        occupied_cells[steps].append(env.starting_pos[r])

    ongoing_frontiers = [None]*nr
    exit_condition = False
    while not env.exploration_grid.all() and not exit_condition and successful_condition:
        if steps > 30:
            breakpoint
        if not env.exploration_grid.all() and all([done[ri] for ri in range(nr)]) and done[ri] == True:
            breakpoint
        steps += 1
        # if steps not in occupied_cells: occupied_cells[steps] = []

        if env.exploration_grid.all():
            break
        # get frontiers
        if any(frontier is None for frontier in ongoing_frontiers):
            start_scheduler_time = time.time()
            frontiers = env.scheduler(ongoing_frontiers)
            end_scheduler_time = time.time()
            k = 0
            for ri in range(nr):
                if frontiers[ri] == env.starting_pos[ri]:
                    k += 1

            if k == nr:
                breakpoint
        
        frontier_list[i].append(frontiers)

        # plan paths
        path_time = 0
        if maneuvering:
            for r in range(nr):
                if env.pos[r] == frontiers[r]:# or done[r]:
                    if env.pos[r] == env.starting_pos[r]:
                        ongoing_frontiers[r] = frontiers[r]
                        done[r] = True
                        which[r] = 0
                        if not env.exploration_grid.all() and all([done[ri] for ri in range(nr)]):
                            save = True
                        if all([done[ri] for ri in range(nr)]) and np.count_nonzero(env.exploration_grid) != WIDTH*HEIGHT and done[ri] == True:
                            breakpoint
                    continue
                
                if ongoing_frontiers[r] == None:
                    start_path_time = time.time()
                    astar = Astar(env.grid)
                    current_path[r], current_maneuvers[r] = astar.a_star(env.pos[r], frontiers[r], env.grid, occupied_cells, env.direction[r])
                    end_path_time = time.time()
                    del current_path[r][0]
                    if maneuvering: del current_maneuvers[r][0]
                
                    # add path to occupied cells
                    for path_step, pos in enumerate(current_path[r]):
                        if steps+path_step not in occupied_cells: occupied_cells[steps+path_step] = []
                        if pos in occupied_cells[steps+path_step]:
                            breakpoint
                        occupied_cells[steps+path_step].append(pos)
                        if len(occupied_cells[steps+path_step]) > nr:
                            breakpoint
        
        # replan paths if drones get stuck
        else:
            for r in range(nr):
                if env.pos[r] == frontiers[r]:# or done[r]:
                    if env.pos[r] == env.starting_pos[r]:
                        ongoing_frontiers[r] = frontiers[r]
                        done[r] = True
                        which[r] = 0
                        if not env.exploration_grid.all() and all([done[ri] for ri in range(nr)]):
                            save = True
                        if all([done[ri] for ri in range(nr)]) and np.count_nonzero(env.exploration_grid) != WIDTH*HEIGHT and done[ri] == True:
                            breakpoint
                    continue
                
            # initialise replanning sequence
            planning = True
            replan_ongoing = False
            prior_r = 0
            r = prior_r
            cntr = 0
            n_plans = 0
            temp_occupied_cells = {key: value[:] for key, value in occupied_cells.items()}
            while planning:
                # loop counters
                if cntr != 0 and cntr % nr == 0:
                    cntr = 0
                    r = prior_r
                if r != 0 and r % nr == 0: r = 0

                # check if drone has on going frontier
                # if there was no path with the on going frontiers path then replan on going frontier paths aswell
                if ongoing_frontiers[r] == None or replan_ongoing:
                    start_path_time = time.time()
                    astar = Astar(env.grid)
                    current_path[r], current_maneuvers[r] = astar.a_star(env.pos[r], frontiers[r], env.grid, temp_occupied_cells, env.direction[r])
                    end_path_time = time.time()

                    # if there is a path add to dynamic obstacles
                    if current_path[r] != None:
                        del current_path[r][0]
                
                        # add path to occupied cells
                        for path_step, pos in enumerate(current_path[r]):
                            if steps+path_step not in temp_occupied_cells: temp_occupied_cells[steps+path_step] = []
                            if pos in temp_occupied_cells[steps+path_step]:
                                breakpoint
                            temp_occupied_cells[steps+path_step].append(pos)
                            
                            if len(temp_occupied_cells[steps+path_step]) > nr:
                                breakpoint
                    else:
                        breakpoint
                
                cntr += 1

                # if all of the paths could be planned 
                if cntr == nr and all([current_path[ri] != None for ri in range(nr)]):
                    planning = False
                    occupied_cells = temp_occupied_cells.copy()
                else:
                    planning = True
                
                # if the current path could not be planned reset priority drone
                if current_path[r] == None:
                    prior_r += 1
                    if prior_r % nr == 0: prior_r = 0
                    n_plans += 1
                    cntr = 0
                    r = prior_r
                    temp_occupied_cells = {key: value[:] for key, value in occupied_cells.items()}
                else:
                    r += 1
                
                # if planned for all drones as priority drone
                if n_plans == nr:
                    # if all of the paths have been replanned including on going paths 
                    if replan_ongoing:
                        print("No route")
                        successful_condition = False
                        unsuccessful += 1
                        planning = False
                    else:
                        # replan all paths again without any ongoing paths
                        replan_ongoing = True
                        prior_r = 0
                        r = prior_r
                        cntr = 0
                        n_plans = 0

                        # delete on oing path fron dynamic obstacles
                        count = 0
                        for key, lst in temp_occupied_cells.items():
                            if len(lst) == 0 or lst[r] is not None:
                                count += 1
                        for c in range(steps, count):
                            del temp_occupied_cells[c]
                            del occupied_cells[c]
            
        if successful_condition:
            for r in range(nr):
                if done[r]:
                    maneuvers[r].append(False)
                    continue
                # execute move 
                trajectory[r].append(current_path[r][0])
                if maneuvering:
                    maneuvers[r].append(current_maneuvers[r][0])
                    if current_maneuvers[r][0] and counter < 20:
                        counter += 1
                        save = True

                # new exploration
                if not env.exploration_grid[current_path[r][0].y, current_path[r][0].x]:
                    explorations[r] += 1

                env.move(r, current_path[r][0])
                
                # check if drone reached frontier
                if env.pos[r] == frontiers[r]:
                    if env.pos[r] == env.starting_pos[r]:
                        ongoing_frontiers[r] = frontiers[r]
                        done[r] = True
                        which[r] = 1
                        if not env.exploration_grid.all() and all([done[ri] for ri in range(nr)]):
                            save = True
                        if all([done[ri] for ri in range(nr)]) and np.count_nonzero(env.exploration_grid) != WIDTH*HEIGHT:
                            breakpoint
                    else:
                        ongoing_frontiers[r] = None
                else:
                    ongoing_frontiers[r] = frontiers[r]

                # remove step from current path
                del current_path[r][0]
                if maneuvering: del current_maneuvers[r][0]

                # add on going frontiers if required
                if len(current_path[r]) != 0 or done[r]:
                    ongoing_frontiers[r] = frontiers[r]
                else:
                    ongoing_frontiers[r] = None

                # add move to actions
                if env.prev_pos[r].x < env.pos[r].x: actions[r].append("right")
                if env.prev_pos[r].x > env.pos[r].x: actions[r].append("left")
                if env.prev_pos[r].y > env.pos[r].y: actions[r].append("up")
                if env.prev_pos[r].y < env.pos[r].y: actions[r].append("down")

                # in loop trajectory drawing
                if in_loop_trajectory and i < saved_iterations:
                    if each_drone:
                        if goal_spawning:
                            env.print_graph(r, steps-1, trajectory[r], maneuvers[r], actions[r], env.starting_pos[r], obstacles, dir_path, i, False, env.goal)
                        else:
                            env.print_graph(r, steps-1, trajectory[r], maneuvers[r], actions[r], env.starting_pos[r], obstacles, dir_path, i, False)
                    else:
                        if r == nr-1:
                            if goal_spawning:
                                env.print_frame(steps, trajectory, maneuvers, actions, env.starting_pos, obstacles, dir_path, i, True, env.goal)
                            else: # [False]+maneuvers[ri] this is because the previous action warns for a maneuver, but since the printing is discrete we only print on the cell where the maneuver happens
                                env.print_frame(steps, trajectory, maneuvers, actions, env.starting_pos, obstacles, dir_path, i, False, None)

                # exit condition
                if goal_spawning and np.array([True for j in range(0, nr) if env.pos[j] == env.goal]).any() == True:
                    exit_condition = True
                
                path_time += end_path_time - start_path_time
            
            schedule_times.append(end_scheduler_time-start_scheduler_time)
            path_times.append(path_time)
            
            # catch maneuvers and save to draw trajectories
            if maneuvering:
                if counter < 20 and not save and env.pos[0] == env.pos[1] and all([not done[ri] for ri in range(nr)]) and any([maneuvers[ri][steps-1] for ri in range(nr)]):
                    save = True
                    step = steps
                    counter += 1
                    breakpoint
                if counter < 20 and not save and env.prev_pos[0] == env.pos[1] and env.prev_pos[1] == env.pos[0] and all([not done[ri] for ri in range(nr)]) and any([maneuvers[ri][steps-1] for ri in range(nr)]):
                    save = True
                    step = steps
                    counter += 1
                    breakpoint
            else:
                if i in saves: save = True

    steps_list.append(steps)
    for ri in range(nr):
        explorations_list[ri].append(explorations[ri])

    if save_trajectory and save or i < saved_iterations:
        maneuvers_list.append(maneuvers)
        for ri in range(nr):
            if goal_spawning:
                env.print_graph(ri, steps, trajectory[ri], maneuvers[ri], actions[ri], env.starting_pos[ri], obstacles, dir_path, i, True, env.goal)
            else: # [False]+maneuvers[ri] this is because the previous action warns for a maneuver, but since the printing is discrete we only print on the cell where the maneuver happens
                env.print_graph(ri, steps, trajectory[ri], [False]+maneuvers[ri], actions[ri], env.starting_pos[ri], obstacles, dir_path, i, True, None)

    # calculate flight time
    flight_time = 0
    flight_distance = 0
    for ri in range(nr):
        for action in actions[ri]:
            t = cell_dimensions / VEL
            flight_distance += cell_dimensions
            flight_time += t
    flight_time = flight_time / nr
    flight_distance = flight_distance / nr
    flight_times.append(flight_time)
    flight_distances.append(flight_distance)

    planning_end_time = time.time()
    planning_time = planning_end_time - planning_starting_time
    planning_times.append(planning_time)

# calculate averages
testing_end_time = time.time()
testing_time = testing_end_time - testing_start_time
tm, ts = divmod(testing_time, 60)
th = 0
if tm >= 60: th, tm = divmod(tm, 60)

average_planning_time = np.mean(np.array(planning_times))
pm, ps = divmod(planning_time, 60)
ph = 0
if pm >= 60: ph, pm = divmod(pm, 60)

average_flight_time = np.mean(np.array(flight_times))
fm, fs = divmod(flight_time, 60)
fh = 0
if fm >= 60: fh, fm = divmod(fm, 60)

average_explorations = []
for drone_explorations in explorations_list:
    # Calculate the average of the current sublist
    drone_average = np.mean(np.array(drone_explorations))
    # Append the average to the corresponding sublist in the averages list
    average_explorations.append(drone_average)

print_string = ""
print_string += "Maneuvering: %s"%(str(maneuvering))
print_string += "\nFixed wing: %s"%(str(fixed_wing))
print_string += "\nFOV width: %dm\nFOV height: %dm" %(FOV_W, FOV_H)
print_string += "\nTesting iterations: %d"%(test_iterations)
print_string += "\nTesting time: %.2fh%.2fm%.2fs" %(th,tm,ts)
print_string += "\nAverage planning time: %.2fh%.2fm%.2fs" %(ph,pm,ps)
print_string += "\nAverage steps: %.2f" %(np.mean(np.array(steps_list)))
print_string += "\nAverage flight time: %.2fh%.2fm%.2fs" %(fh,fm,fs)
print_string += "\nAverage flight distance: %.2f m" %(np.mean(np.array(flight_distances)))
for ri in range(nr):
    print_string += "\nAverage explorations for drone %d: %.2f" %(ri, average_explorations[ri])
print_string += "\nAverage time scheduling: %.8fs"%(np.mean(np.array(schedule_times)))
print_string += "\nAverage time path planning: %.8fs"%(np.mean(np.array(path_times)))
print_string += "\nPercentage success: %.2f"%((test_iterations-unsuccessful)/test_iterations*100)
print_string += "\nObstacles: %.2f"%(obstacle_density)
print(print_string)

file_name = "results.txt"
file_path = os.path.join(dir_path, file_name)
with open(file_path, 'w') as file:
    file.write(print_string)

file_name = "frontiers.txt"
file_path = os.path.join(dir_path, file_name)
with open(file_path, 'w') as file:
    # Iterate through the sublists and write them to the file
    for sublist in frontier_list:
        # Convert sublist elements to strings and join them with commas
        sublist_str = ','.join(map(str, sublist))
        # Write the sublist string followed by a newline character
        file.write(sublist_str + '\n')

file_name = "maneuvers.txt"
file_path = os.path.join(dir_path, file_name)
with open(file_path, 'w') as file:
    # Iterate through the sublists and write them to the file
    for sublist in maneuvers_list:
        # Convert sublist elements to strings and join them with commas
        sublist_str = ','.join(map(str, sublist))
        # Write the sublist string followed by a newline character
        file.write(sublist_str + '\n')

if save_obstacles:
    # save starting positions
    # for my solution
    file_name = "positions.pkl"
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, 'wb') as file:
        # Serialize and write the sublists using pickle.dump()
        pickle.dump(env.starting_positions, file)

    # for DARP
    starting_positions = convert_coordinates_to_cells(env.starting_positions)
    file_name = "positions.json"
    file_name = os.path.join(save_dir, file_name)
    write_json(starting_positions, file_name)

    # save target positions
    # for my solution
    file_name = "targets.pkl"
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, 'wb') as file:
        # Serialize and write the sublists using pickle.dump()
        pickle.dump(env.goals, file)

    # for DARP
    goals = convert_target_coordinates_to_cells(env.goals)
    file_name = "targets.json"
    file_name = os.path.join(save_dir, file_name)
    write_json(goals, file_name)

    # save grids
    # for my solution
    file_name = "grid.json"
    file_name = os.path.join(save_dir, file_name)
    write_json(env.grids, file_name)

    # for DARP
    obstacle_positions = convert_grid_to_obstacle_positions(env.grids)
    file_name = "obstacles.json"
    file_name = os.path.join(save_dir, file_name)
    write_json(obstacle_positions, file_name)