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
from functools import reduce

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

        return temp_grid[y,x] != States.OBS.value

    def neighbors(self, id, path_step, direction):
        (x, y) = id

        # (right, up, left, down)
        results = [Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        
        # This is done to prioritise straight paths
        #if (x + y) % 2 == 0: results.reverse()
        results = list(filter(self.in_bounds, results))
        results = list(filter(lambda k: self.is_obstacle(k, id, path_step), results))
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
        while (id,current) != (0,start):
            # for debugging
            if (id,current) not in self.came_from:
                print((id,current) ,self.came_from)
                print(self.grid, self.start, end, "this one")

            new_id, new_current = self.came_from[(id, current)]
            current = new_current
            id = new_id
            path.append(current)
        path.reverse()
        return path

    # A* algorithm
    def a_star(self, start, end, grid, dynamic_obstacles, direction=None):
        self.grid = grid # grid for debugging
        self.dynamic_obstacles = dynamic_obstacles # tracks dynamic obstacles (drones)
        self.graph = Grid(HEIGHT, WIDTH, grid, direction) # up to date environment without dynamic obstacles
        self.start = start # starting position
        self.came_from = {} # tree
        self.cost_so_far = {} # closed set
        self.heap = [(0, 0, start, direction)] # open set
        
        self.cost_so_far[(0,start)] = 0 # initialise the costs
        current = start
        found = False
        id = 0
        
        while len(self.heap) > 0:
            _, id, current, direction = heapq.heappop(self.heap) # pop open set
            path_step = id
            # if the current cell is equal to the end cell
            if current == end:
                found = True
                break
            # for debugging
            if id > HEIGHT*WIDTH:
                breakpoint
                print("Maybe stuck...")
            
            self.neighbors = self.graph.neighbors(current, path_step, direction) # get neighbours
            for next_node in self.neighbors:
                # on location collision
                # if a drone has planned thus far
                # AND a drone is on next_node location
                if steps+path_step in self.dynamic_obstacles and next_node in self.dynamic_obstacles[steps+path_step]:
                    continue
                # cross location collision
                # current is in dynamic obstacles on next step
                if steps+path_step in self.dynamic_obstacles and current in self.dynamic_obstacles[steps+path_step]:
                    # next node is in dynamic obstacles on previous step
                    if steps+path_step-1 in self.dynamic_obstacles and next_node in self.dynamic_obstacles[steps+path_step-1]:
                        continue

                new_cost = self.cost_so_far[(id,current)] + self.heuristic(current, next_node)

                # if next node and next id is not in the closed set
                # add to closed and open sets
                if (id+1,next_node) not in self.cost_so_far:
                    self.cost_so_far[(id+1,next_node)] = new_cost
                    priority = new_cost + self.heuristic(next_node, end)
                    heapq.heappush(self.heap, (priority, id+1, next_node, direction))
                    self.came_from[(id+1,next_node)] = (id, current)
        
        # if current cell is equal to end cell
        # then the end cell is found
        if current == end:
            found = True
        # if current != end:
        #     print("bugga%d"%(i))
            
        # if the end cell was found
        # then reconstruct path
        # else return nothing
        if found:
            self.which = 1
            return self.reconstruct_path(start, end, id)
        else:
            return None
    
class Environment:
    def __init__(self, nr, obstacles, set_obstacles, obstacle_density, save_obstacles, save_dir, load_obstacles, load_dir):
        # initalise variables
        self.nr = nr
        self.obstacles = obstacles # turn obstacles on or off
        self.set_obstacles = set_obstacles # set obstacles to random or static
        self.obstacle_density = obstacle_density # sets density
        self.save_obstacles = save_obstacles # sets if obstacle layout is saved
        self.save_dir = save_dir # points to path to save
        self.load_obstacles = load_obstacles # sets if obstacle layout should be loaded from previous simulation
        self.load_dir = load_dir #  points to path to load from  
        self.distances = [] # list of distances to each cell
        # for debugging
        self.first = []
        self.second = []
        self.third = []
        self.fourth = []
        self.fifth = []
        self.fifth_a = []
        self.fifth_b = []
        self.fifth_b_1 = []
        self.fifth_b_2 = []
        self.sixth = []
        self.seventh = []
        self.first_dist = []
        self.second_dist = []
        self.thrid_dist = []
        self.fourth_dist = []
        self.fifth_dist = []
        
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

    def reset(self, goal_spawning, i=0):
        # spawn grid
        global WIDTH
        global HEIGHT
        global distances
        global temp_dist_vec
        global trajectory
        self.grid = np.zeros((HEIGHT, WIDTH))

        # generates obstacles
        if self.obstacles:
            if self.load_obstacles:
                self.starting_grid = np.array(self.grids[i])
                self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))
                self.grid = self.starting_grid.copy()
            elif not self.set_obstacles:
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

                WIDTH = int(WIDTH/2)*2
                HEIGHT = int(HEIGHT/2)*2

                self.ES = Enclosed_space_check(int(HEIGHT/2), int(WIDTH/2), self.starting_grid, States)
                self.ES_starting_grid = self.ES.enclosed_space_handler()

                self.starting_grid = np.kron(self.ES_starting_grid, np.ones((2, 2)))
                self.grid = self.starting_grid.copy()

        self.grid = self.starting_grid.copy()
        
        # spawn drone
        if self.load_obstacles and self.starting_positions[0] != []: # load existing positions
            self.starting_pos = [Point(self.starting_positions[i][ri].x*2, self.starting_positions[i][ri].y*2) for ri in range(nr)]
            for ri in range(self.nr):
                self.pos[ri] = self.starting_pos[ri]
                self.prev_pos[ri] = self.starting_pos[ri]
                self.grid[self.starting_pos[ri].y, self.starting_pos[ri].x] = States.ROBOT.value
        else: # generate new positions
            if self.set_obstacles:
                indices = np.argwhere(np.array(self.ES_starting_grid) == States.UNEXP.value)
            else:
                indices = np.argwhere(np.array(self.ES_starting_grid) == States.UNEXP.value)
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
                    indices = np.argwhere(np.array(self.ES_starting_grid) == States.UNEXP.value)
                else:
                    indices = np.argwhere(np.array(self.ES_starting_grid) == States.UNEXP.value)
                np.random.shuffle(indices)
                save_goal_pos = Point(indices[0,1], indices[0,0])
                self.goal = Point(indices[0,1]*2, indices[0,0]*2)
            if self.save_obstacles:
                self.goals[i] = save_goal_pos

        if self.save_obstacles:
            self.starting_positions[i] = save_starting_pos.copy()
            self.grids.append(self.ES_starting_grid.tolist())

        # initialise exploration grid
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[y,x] == States.OBS.value: self.exploration_grid[y, x] = True

        # set target cluster
        self.target_cluster = [[None]]*self.nr

        distances = np.full((HEIGHT,WIDTH, HEIGHT,WIDTH), HEIGHT * WIDTH)  # Initialize distances to HEIGHT*WIDTH (unreachable)
        for ri in range(self.nr):
            distances[self.pos[ri].y][self.pos[ri].x][self.pos[ri].y][self.pos[ri].x] = 0
        temp_dist_vec = np.copy(distances)
        self.neighbours = {}
        self.neighbours_complete = {}
        trajectory = [[self.starting_pos[r]] for r in range(nr)]
        self.known_cells = [np.array([n.y, n.x]) for n in self.pos]
        env.calculate_distances()

        # set starting pos to true in exploration grid
        for r in range(self.nr): self.exploration_grid[self.pos[r].y, self.pos[r].x] = True
        # initialise explored cells
        self.explorated_cells = []
        for ri in range(self.nr):
            self.explorated_cells.append((self.starting_pos[ri].y,self.starting_pos[ri].x))
        
    def move(self, r, new_pos):   
        # move drone to new position
        self.prev_pos[r] = self.pos[r]
        self.pos[r] = Point(new_pos.x,new_pos.y)
        
        # update grids
        self.grid[self.prev_pos[r].y, self.prev_pos[r].x] = States.EXP.value
        self.grid[self.pos[r].y, self.pos[r].x] = States.ROBOT.value

        if self.prev_pos[r].x < self.pos[r].x: # right
            self.direction[r] = "right"
        elif self.prev_pos[r].x > self.pos[r].x: # left
            self.direction[r] = "left"
        elif self.prev_pos[r].y < self.pos[r].y: # down
            self.direction[r] = "down"
        elif self.prev_pos[r].y > self.pos[r].y: # up
            self.direction[r] = "up"

        self.calculate_distances()
        
        self.exploration_grid[self.prev_pos[r].y, self.prev_pos[r].x] = True
        self.exploration_grid[self.pos[r].y, self.pos[r].x] = True
        self.explorated_cells.append((self.pos[r].y,self.pos[r].x))
        
    def get_min_targets(self, costs):
        # get min distance of each dorne
        min_targets_value = [None]*self.nr
        for ri in range(self.nr):
            if not costs[ri]: min_targets_value[ri] = None
            else:
                min_value = min(costs[ri].values())
                min_targets_value[ri] = min_value

        # get all positions distances equal min distance
        min_targets = [[] for i in range(self.nr)]
        for ri in range(self.nr):
            if min_targets_value[ri] == None: continue
            min_targets[ri] = [key for key, value in costs[ri].items() if costs[ri] if value == min_targets_value[ri] ]
            
        return min_targets
        
    def scheduler(self, ongoing_frontiers):
        start = time.time()
        # set current positions to explored
        temp_exploration_grid = self.exploration_grid.copy()
        for ri in range(self.nr): 
            temp_exploration_grid[self.pos[ri].y, self.pos[ri].x] = True

        # if no more unexplored cells
        # then return to home
        if np.count_nonzero(temp_exploration_grid) == HEIGHT*WIDTH:
            return [self.starting_pos[ri] for ri in range(self.nr)]
        
        end = time.time()
        self.first.append(end-start)
        start = time.time()

        
        for ri in range(nr):
            if ongoing_frontiers[ri] == None or return_home[ri]: continue
            # if on going candidate already searched
            # then set on going condidate to None
            if temp_exploration_grid[ongoing_frontiers[ri].y, ongoing_frontiers[ri].x]:
                ongoing_frontiers[ri] = None

                # delete on going path from dynamic obstacles
                count = list(occupied_cells)[-1]
                for c in range(steps, count+1):
                    # checks that the current path of the drone is not shorter than the occupied cells dictionary
                    if len(current_path[ri]) == c-steps: break
                    occupied_cells[c].remove(current_path[ri][c-steps])
                    if len(occupied_cells[c]) == 0:
                        del occupied_cells[c]
                current_path[ri] = []

        # for testing
        end = time.time()
        self.second.append(end-start)
        start = time.time()

        # get costs of unexplored cells
        # TODO do without for loops
        costs = [{} for _ in range(self.nr)]
        for ri in range(self.nr):
            # if ongoing_frontiers[ri] != None: continue
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    if not temp_exploration_grid[y,x]:
                        # costs[ri][Point(x,y)] = distances[self.pos[ri]][Point(x,y)]
                        costs[ri][Point(x,y)] = distances[self.pos[ri].y][self.pos[ri].x][y][x]

        # for testing
        end = time.time()
        self.third.append(end-start)
        start = time.time()

        # get minimum targets
        # process:
        # get the minimum distance
        # delete minimum from cost list
        # repeat for number of drones minus 1
        # it's repeated this many times since drones could share the minumum distance cells
        # to allow each drone to have an option the process is repeated n-1 times to give each drone at least 1 option
        # this becomes important when there are only n cells left to explore
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
        
        # for testing
        end = time.time()
        self.fourth.append(end-start)
        start = time.time()

        # get all combinations of best targets
        combinations = list(product(*min_targets))

        # for testing
        end_a = time.time()
        self.fifth_a.append(end_a-start)
        start_b_1 = time.time()

        # find invalid combinations
        delete_indices = []
        for i,combination in enumerate(combinations):
            # if the combination has any duplicates
            # then remove from all combinations
            if len(combination) != len(set(combination)):
                delete_indices.append(i)
            # check drone neighbours
            # if drone only has 1 valid neighbour mark node as invalid for other drones
            # TODO

        # for testing
        end_b_1 = time.time()
        self.fifth_b_1.append(end_b_1-start_b_1)
        start_b_2 = time.time()
        
        # delete invalid combinations
        delete_indices_set = set(delete_indices)
        modified_combinations = [combination for i, combination in enumerate(combinations) if i not in delete_indices_set]
        combinations = modified_combinations.copy()

        # for testing
        end = time.time()
        self.fifth.append(end-start)
        self.fifth_b.append(end-start_b_1)
        self.fifth_b_2.append(end-start_b_2)

        # if there are valid combinations
        # then find best combination and return
        targets = [None]*self.nr
        if len(combinations) > 0:
            start = time.time()
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

            # for debugging
            if not self.exploration_grid.all() and all([targets[ri] == self.starting_pos[ri] for ri in range(self.nr)]) and targets[0] == self.starting_pos[0]:
                breakpoint

            # for testing
            end = time.time()
            self.sixth.append(end-start)

            return targets

        # for testing
        start = time.time()
        
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

        # for debugging
        if not self.exploration_grid.all() and all([targets[ri] == self.starting_pos[ri] for ri in range(self.nr)]) and targets[0] == self.starting_pos[0]:
            breakpoint

        # for testing
        end = time.time()
        self.seventh.append(end-start)

        return targets
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < WIDTH and 0 <= y < HEIGHT
    
    def is_obstcle(self, id):
        (x, y) = id
        return env.grid[y,x] != States.OBS.value

    def is_known(self, id):
        (x, y) = id
        return np.any(np.all(np.array([y,x]) == np.array(self.known_cells), axis=1))
    
    def calculate_distances(self):
        global distances
        cells_to_update = []
        for ri in range(self.nr):
            # Add curretn position to cells to update
            cells_to_update.append(np.array([self.pos[ri].y, self.pos[ri].x]))

            # if the current position is not in neighbours collection
            # then get neighbours and add to neighbours
            # BUG!!!!!!!!!!!!!!!!
            # if not explored
            # add neighbours as parents and add all known neighbours
            start = time.time()
            if not self.exploration_grid[self.pos[ri].y, self.pos[ri].x]:
                # get neighbours
                # (right, up, left, down)
                results = [[self.pos[ri].x+1, self.pos[ri].y],
                        [self.pos[ri].x, self.pos[ri].y-1],
                        [self.pos[ri].x-1, self.pos[ri].y],
                        [self.pos[ri].x, self.pos[ri].y+1]]
                results = list(filter(self.in_bounds, results))
                results = list(filter(self.is_obstcle, results))
            
                self.neighbours[self.pos[ri]] = np.array(results)
                # Update neighbours of the neighbours
                for neighbour in results:
                    # Add to known list
                    if not np.any(np.all(np.array([neighbour[1],neighbour[0]]) == np.array(self.known_cells), axis=1)): self.known_cells.append(np.array([neighbour[1],neighbour[0]]))

                    # If neighbour is not in neighbours dictionary
                    # Then add new key to dictionary with current position as neighbour
                    if Point(neighbour[0], neighbour[1]) not in self.neighbours: self.neighbours[Point(neighbour[0], neighbour[1])] = [np.array([self.pos[ri].x, self.pos[ri].y])]
                    # If neighbour does not have the current position as neighbour, but it's already in the neighbours dictionary
                    # Then add current position to neigbours
                    elif not np.any(np.all(np.array([self.pos[ri].x, self.pos[ri].y]) == self.neighbours[Point(neighbour[0], neighbour[1])], axis=1)):
                        np.append(self.neighbours[Point(neighbour[0], neighbour[1])], np.array([self.pos[ri].x, self.pos[ri].y]))

                    # Add neighbour and neighbours known second neighbours to update list
                    if not np.any(np.all(np.array([neighbour[1],neighbour[0]]) == np.array(cells_to_update), axis=1)):
                        cells_to_update.append(np.array([neighbour[1],neighbour[0]]))
                    # get second neighbours
                    second_neighbours = [
                                        [neighbour[0]+1, neighbour[1]],
                                        [neighbour[0], neighbour[1]-1],
                                        [neighbour[0]-1, neighbour[1]],
                                        [neighbour[0], neighbour[1]+1]
                                        ]
                    # remove any invalid neighbours (out of bounds or not known) and add to neighbours dictionary
                    second_neighbours = list(filter(self.in_bounds, second_neighbours))
                    neighbours = list(filter(self.is_known, second_neighbours))
                    self.neighbours[Point(neighbour[0],neighbour[1])] = np.array(neighbours)

                    # Mask the already calculated distances to avoid overwriting them
                    mask = distances[
                            neighbour[1],
                            neighbour[0],
                            self.neighbours[Point(neighbour[0],neighbour[1])][:, 1], 
                            self.neighbours[Point(neighbour[0],neighbour[1])][:, 0]] \
                            == HEIGHT*WIDTH
                    # Update distance from current position to unknown neighbours
                    distances[
                        neighbour[1],
                        neighbour[0],
                        self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 1],
                        self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 0]] \
                        = 1
                    # Update distance from unknown neighbours to unknown neighbours
                    distances[
                        self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 1], 
                        self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 0], 
                            self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 1], 
                            self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 0]] \
                            = 0
                    # Update distance from unknown neighbours to current position
                    distances[
                        self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 1], 
                        self.neighbours[Point(neighbour[0],neighbour[1])][mask][:, 0], 
                        neighbour[1], 
                        neighbour[0]] \
                        = 1

                self.neighbours_complete[self.pos[ri]] = True
                
            # Mask the already calculated distances to avoid overwriting them
            mask = distances[self.pos[ri].y, self.pos[ri].x, \
                            self.neighbours[self.pos[ri]][:, 1], self.neighbours[self.pos[ri]][:, 0]] == HEIGHT*WIDTH
            # Update distance from current position to unknown neighbours
            distances[self.pos[ri].y, self.pos[ri].x, \
                      self.neighbours[self.pos[ri]][mask][:, 1], self.neighbours[self.pos[ri]][mask][:, 0]] = 1
            # Update distance from unknown neighbours to unknown neighbours
            distances[self.neighbours[self.pos[ri]][mask][:, 1], self.neighbours[self.pos[ri]][mask][:, 0], \
                      self.neighbours[self.pos[ri]][mask][:, 1], self.neighbours[self.pos[ri]][mask][:, 0]] = 0
            # Update distance from unknown neighbours to current position
            distances[self.neighbours[self.pos[ri]][mask][:, 1], self.neighbours[self.pos[ri]][mask][:, 0], \
                      self.pos[ri].y, self.pos[ri].x] = 1
            end = time.time()
            self.first_dist.append(end-start)

        # # Update any new neighbours here
        # # Set mask of all known cells
        # start = time.time()
        # fmask = []
        # for pos in self.pos:
        #     fmask_matrix = distances[pos.y, pos.x] != WIDTH * HEIGHT
        #     temp_fmask = np.transpose(np.where(fmask_matrix)).tolist()
        #     temp_fmask_set = set(map(tuple, temp_fmask))
        #     # Accumulate fmask_set
        #     fmask.append(temp_fmask)
        # fmask = np.array([item for row in fmask for item in row])
        # breakpoint

        # # Update neighbours of all known cells
        # fmask_set = set(map(tuple, fmask))
        # to_be_removed = []
        # for cell_i, cell in enumerate(fmask):
        #     if Point(cell[1],cell[0]) in self.neighbours_complete: continue
        #     results = [[cell[1]+1, cell[0]],
        #             [cell[1], cell[0]-1],
        #             [cell[1]-1, cell[0]],
        #             [cell[1], cell[0]+1]]
        #     results = list(filter(self.in_bounds, results))
        #     neighbours = [n for n in results if tuple((n[1],n[0])) in fmask_set]
        #     self.neighbours[Point(cell[1],cell[0])] = np.array(neighbours)

        #     # Mask the already calculated distances to avoid overwriting them
        #     mask = distances[
        #             cell[0],
        #             cell[1],
        #             self.neighbours[Point(cell[1],cell[0])][:, 1], 
        #             self.neighbours[Point(cell[1],cell[0])][:, 0]] \
        #             == HEIGHT*WIDTH
        #     # Update distance from current position to unknown neighbours
        #     distances[
        #         cell[0],
        #         cell[1],
        #         self.neighbours[Point(cell[1],cell[0])][mask][:, 1],
        #         self.neighbours[Point(cell[1],cell[0])][mask][:, 0]] \
        #         = 1
        #     # Update distance from unknown neighbours to unknown neighbours
        #     distances[
        #         self.neighbours[Point(cell[1],cell[0])][mask][:, 1], 
        #         self.neighbours[Point(cell[1],cell[0])][mask][:, 0], 
        #             self.neighbours[Point(cell[1],cell[0])][mask][:, 1], 
        #             self.neighbours[Point(cell[1],cell[0])][mask][:, 0]] \
        #             = 0
        #     # Update distance from unknown neighbours to current position
        #     distances[
        #         self.neighbours[Point(cell[1],cell[0])][mask][:, 1], 
        #         self.neighbours[Point(cell[1],cell[0])][mask][:, 0], 
        #         cell[0], 
        #         cell[1]] \
        #         = 1
                
            # self.grid_plot()
            # plt.show()
            # breakpoint
            end = time.time()
            self.second_dist.append(end-start)

        np_known_cells = np.array(self.known_cells)
        for ri in range(self.nr):
            start = time.time()
            # Set neighbours of neighbours
            c_neighbours = [Point(neighbour[0], neighbour[1]) for neighbour in self.neighbours[self.pos[ri]]]
            n_neighbours = [self.neighbours[key] for key in c_neighbours]
            n_neighbours.append(np.array([[self.pos[ri].x, self.pos[ri].y]]))
            n_neighbours = np.array([item for row in n_neighbours for item in row])
            end = time.time()
            self.thrid_dist.append(end-start)
            breakpoint
            start = time.time()

            # update neighbours
            for cell in c_neighbours:
                cell = np.array([cell.y, cell.x])
                distances[
                    np.repeat(cell[0], np_known_cells[:,0].shape[0]),
                    np.repeat(cell[1], np_known_cells[:,1].shape[0]),
                    np_known_cells[:, 0],
                    np_known_cells[:, 1]
                    ] \
                =\
                np.minimum(\
                    # From cell to all known cells
                    distances[ 
                        np.repeat(cell[0], np_known_cells[:,0].shape[0]),
                        np.repeat(cell[1], np_known_cells[:,1].shape[0]),
                        np_known_cells[:, 0],
                        np_known_cells[:, 1]
                    ],
                    # 
                    np.minimum.reduceat(
                        distances[ 
                            np.tile(np_known_cells[:, 0], np_known_cells[:,0].shape[0]),
                            np.tile(np_known_cells[:, 1], np_known_cells[:,0].shape[0]),
                            np.tile(np.repeat(cell[0], np_known_cells[:,0].shape[0]), np_known_cells[:,0].shape[0]),
                            np.tile(np.repeat(cell[1], np_known_cells[:,1].shape[0]), np_known_cells[:,0].shape[0])
                        ]\
                        +\
                        distances[ 
                            np.tile(np_known_cells[:,0], np_known_cells[:,0].shape[0]),
                            np.tile(np_known_cells[:,1], np_known_cells[:,0].shape[0]),
                            np.repeat(np_known_cells[:,0], np_known_cells[:,0].shape[0]),
                            np.repeat(np_known_cells[:,1], np_known_cells[:,0].shape[0])                        
                            ],
                        np.arange(0, np_known_cells[:,0].shape[0]*np_known_cells[:,0].shape[0], np_known_cells[:,0].shape[0])
                    )
                        
                    )
                # self.grid_plot()
                # plt.show()
                # breakpoint
                # plt.close()
                
            end = time.time()
            self.fourth_dist.append(end-start)
            
            # self.grid_plot()
            # plt.show()
            # breakpoint
            start = time.time()
            
            # Update mirrored distances of updated neighbour distances
            distances[
                np.tile(np_known_cells[:, 0],self.neighbours[self.pos[ri]][:, 1].shape[0]),
                np.tile(np_known_cells[:, 1],self.neighbours[self.pos[ri]][:, 0].shape[0]),
                np.repeat(self.neighbours[self.pos[ri]][:, 1], np_known_cells[:,0].shape[0]),
                np.repeat(self.neighbours[self.pos[ri]][:, 0], np_known_cells[:,1].shape[0])                
                ]\
            =\
            distances[
                np.repeat(self.neighbours[self.pos[ri]][:, 1], np_known_cells[:,0].shape[0]),
                np.repeat(self.neighbours[self.pos[ri]][:, 0], np_known_cells[:,1].shape[0]),
                np.tile(np_known_cells[:, 0],self.neighbours[self.pos[ri]][:, 1].shape[0]),
                np.tile(np_known_cells[:, 1],self.neighbours[self.pos[ri]][:, 0].shape[0])
                ]
            
            end = time.time()
            self.fifth_dist.append(end-start)
            
        
        # self.grid_plot()
        # # self.grid_plot_vec()
        # plt.show()
        # breakpoint
        # plt.close()

                    
    def grid_plot(self):
        grid_size = int(math.sqrt(HEIGHT*WIDTH))
        # Create a figure and axes
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(8, 8))

        # Loop through each cell of the larger grid
        for j in range(HEIGHT):
            for i in range(WIDTH):
                # Create a subplot for the current cell
                ax = axs[j, i]

                # Plot the smaller grid with actual values
                ax.imshow(distances[j][i], cmap='viridis')

                for y in range(distances[j][i].shape[0]):
                    for x in range(distances[j][i].shape[1]):
                        ax.text(x, y, f'{int(distances[j, i, y, x])}', va='center', ha='center', color='white', fontsize=6)

                # Optionally, you can set titles for each subplot
                ax.set_title(f'Subgrid {j * grid_size + i + 1}')

                # Remove axis ticks and labels if desired
                ax.axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()

    def grid_plot_vec(self):
        grid_size = int(math.sqrt(HEIGHT*WIDTH))
        # Create a figure and axes
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(8, 8))

        # Loop through each cell of the larger grid
        for j in range(HEIGHT):
            for i in range(WIDTH):
                # Create a subplot for the current cell
                ax = axs[j, i]

                # Plot the smaller grid with actual values
                ax.imshow(temp_dist_vec[j][i], cmap='viridis')

                for y in range(temp_dist_vec[j][i].shape[0]):
                    for x in range(temp_dist_vec[j][i].shape[1]):
                        ax.text(x, y, f'{int(temp_dist_vec[j, i, y, x])}', va='center', ha='center', color='white')

                # Optionally, you can set titles for each subplot
                ax.set_title(f'Subgrid {j * grid_size + i + 1}')

                # Remove axis ticks and labels if desired
                ax.axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
    
    def print_graph(self, r, steps, path, actions, starting_pos, obstacles, dir_path, cnt, summary=False, goal_pos=None, step=None):
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
            elif goal_pos != Point(x,y) and Point(x,y) != starting_pos:
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                        [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                        facecolor="blue", 
                        alpha=0.5)
            
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

    def print_frame(self, steps, path, actions, starting_pos, obstacles, dir_path, cnt, summary=False, goal_pos=None, step=None):
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
                elif goal_pos != Point(x,y) and Point(x,y) != starting_pos:
                    ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                            [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                            facecolor="blue", 
                            alpha=0.5)
            
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
        # plt.pause(0.0005)
        plt.close()

##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
# Initialisations
# Simulation initialisations
test_iterations = 1 # Number of simulation iterations
goal_spawning = False # Sets exit condition: finding the goal or 100% coverage

# Environment initialisations
nr = 2 # number of drones
obstacles = True # Sets of obstacels spawn
obstacle_density = 0 # Sets obstacle density      <---------------------------------------------------------------------- (set obstacles variable to be automatic with 0 density)
set_obstacles = False # Sets if obstacles should change each iteration
save_obstacles = True # Sets if obstacles are saved
load_obstacles = False # Sets if obstacles should be loaded from previous simulation

# Trajectory saving initialisations
save_trajectory = True # Sets if drone trajectories are saves
step_trajectory = True # Sets if the drone trajectories are saves each step
each_drone = False # Stes if drone trajectories are saved separately in the step trajectories
saved_iterations = 2 # Sets number of iteration trajectories are saved
saves = []                                     #    <---------------------------------------------------------------------- (IDK what this does)

# Directory initialisations
PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')

# Set directory path
if save_trajectory:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'Astar')      

    date_and_time = datetime.now()
    dir_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(dir_path): os.makedirs(dir_path)

    # Set save path
    save_dir = os.path.join(dir_path, 'Save')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

# Set load path
load_dir = os.path.join(PATH, 'Load') 
if not os.path.exists(load_dir): os.makedirs(load_dir)

# Tracking initialisations
testing_start_time = time.time() # starts simulation testing time
steps_list = [] # tracks number of steps per iteration
explorations_list = [[] for r in range(nr)] # tracks the number of cells each drone explores during an iteration
planning_times = [] # tracks time per iteration
flight_times = [] # tracks flight time per iteration
flight_distances = [] # tracks flight distance per ietration
schedule_times = [] # tracks each scheduling time 
dist_update_times = []
path_times = [] # track each path planning time
frontier_list = [[] for _ in range(test_iterations)] # tracks scheduled candidates for each iteration in separate lists
unsuccessful = 0 # tracks number of unsuccessful search attempts 
sheduled_distances = [[] for _ in range(nr)] # tracks the distance a candidate cell is from the current cell
trajectory = [[] for r in range(nr)] # tracks drone trajectories

# Environment generation
env = Environment(nr, obstacles, set_obstacles, obstacle_density, save_obstacles, save_dir, load_obstacles, load_dir)
env.reset(goal_spawning)
print(env.ES_starting_grid)

# Simulation testing loop
for i in range(test_iterations):
    # Iteration initialisations
    planning_successful = True # boolean for checking if schedule and path planning is successful
    save = False #                              #    <---------------------------------------------------------------------- (IDK what this does)
    planning_starting_time = time.time() # sets starts time for current iteration
    if i % 500 == 0: print(i) # prints every x iterations
    # the first iteration has already been reset
    # thus it does not have to be run again
    if i != 0:
        env.reset(goal_spawning, i)
    obstacle_layout = env.exploration_grid.copy() # set obstacle layout for iteration
    steps = 0 # sets number of steps equal to zero
    actions = [[] for _ in range(nr)] # tracks actions for current iteration
    return_home = [False for _ in range(nr)] # boolean for tracking drone state. If True the drone returns to starting position 
    current_path = [[] for _ in range(nr)] # tracks current path of drone
    finished_scheduling = [False]*nr # boolean for tracking scheduling state of drone. If True the drone returns to starting position #    <---------------------------------------------------------------------- (check if same functionality as return home state)
    # trajectory = [[env.starting_pos[r]] for r in range(nr)] # tracks drone trajectories
    which = [False]*nr # for debugging
    r_which = [] # for debugging
    occupied_cells = {} # tracks occupancy state of cells at specified steps
    occupied_cells[steps] = [] # initialises the list occupancy states for step zero
    explorations = [0]*nr # initialises number of cells explored for each drone for this iteration
    # initialises the occupancy states for step zero
    for r in range(nr):
        occupied_cells[steps].append(env.starting_pos[r])

    ongoing_frontiers = [None]*nr # initialises on going candidates (this is a variable which checks if the drone has reach its candidate cell)
    goal_exit_condition = False # initialises boolean for exit condition (checks if any drone has reached goal)
    # loop for planning until an exit condition is met
    while not env.exploration_grid.all() and not goal_exit_condition and planning_successful:
        if steps > 1000:
            breakpoint
        if not env.exploration_grid.all() and all([finished_scheduling[ri] for ri in range(nr)]) and finished_scheduling[ri] == True:
            breakpoint

        steps += 1 # counts number of steps
        
        # if all cells have been explored then exit loop
        if env.exploration_grid.all():
            break

        # get frontiers
        temp_ongoing_frontiers = ongoing_frontiers.copy() # save current on going candidates
        # if any of the drones do not have on going candidates
        # then schedule candidates
        if any(frontier is None for frontier in ongoing_frontiers):
            start_scheduler_time = time.time() # tracks scheduling time
            frontiers = env.scheduler(ongoing_frontiers) # schedules candidates
            end_scheduler_time = time.time() # tracks scheduling time
            
            # checks if any drones have its starting positions as its candidate
            # if True then there are not any better cells to travel to, which means the area is almost covered
            for ri in range(nr):
                if frontiers[ri] == env.starting_pos[ri]:
                    return_home[ri] = True
                else:
                    return_home[ri] = False
        
        frontier_list[i].append(frontiers) # tracks scheduled candidates of drones

        # plan paths
        path_time = 0 # tracks path planning time
        # if drones are returning home and if the drone has been taken out of consideration for scheduling 
        for r in range(nr):
            # if drone is at candidate and the candidate is the dtarting postition
            if env.pos[r] == frontiers[r] and env.pos[r] == env.starting_pos[r]:# or done[r]:
                # if the drone is returning home and not already taken out of consideration for scheduling
                # then flag drone as finished scheduling
                if return_home[r] and not finished_scheduling[r]: #    <---------------------------------------------------------------------- (is it possible to not enter this statement?)
                    finished_scheduling[r] = True # flags drone as finished searching (at home/landed)
                    which[r] = 0 # for debugging
                    if not env.exploration_grid.all() and all([finished_scheduling[ri] for ri in range(nr)]): #    <---------------------------------------------------------------------- (i think old functionality)
                        save = True
                    if all([finished_scheduling[ri] for ri in range(nr)]) and np.count_nonzero(env.exploration_grid) != WIDTH*HEIGHT and finished_scheduling[ri] == True:
                        breakpoint
                    ongoing_frontiers[r] = frontiers[r] # sets on going candidate as current candidate (which is starting position) this way the drone will not be scheduled another candidate 
                continue
            
        # initialise replanning sequence
        planning = True # boolean for tracking planning state
        replan_ongoing = False # boolean for tracking re-planning. if a path plan failed the algorithm has to re-plan the paths
        prior_r = 0 # sets the planning priority of the drones
        r = prior_r # sets the priority drone to the current drone
        cntr = 0 # initialises the number of plans for current drone as priority drone to zeros
        n_plans = 0 # initialises the number of plans
        temp_occupied_cells = {key: value[:] for key, value in occupied_cells.items()} # sets the occupancy of all cells to current history

        # loop for path planning
        while planning:
            # if the number of plans for current priority drone is not equal to zero
            # AND (the number of plans for current priorty drone) MOD (the number of drones) equals zero
            # then the current priority drone has to be changed since no planned path was successfull
            if cntr != 0 and cntr % nr == 0:
                cntr = 0 # sets the number of plans for current drone as priority drone to zeros
                r = prior_r # sets the selected drone to the priority drone

            # if the selected drone is not equal to zero
            # AND (the selected drone) MOD (the number of drones) equals zero
            # then set selected drone to first drone
            if r != 0 and r % nr == 0: r = 0

            # if drone does not have an on going candidate
            # OR there was no path with the on going frontiers path
            # then replan on going frontier paths aswell
            if ongoing_frontiers[r] == None or replan_ongoing:
                start_path_time = time.time() # sets start time for path planning
                astar = Astar(env.grid) # initialises A* class
                current_path[r] = astar.a_star(env.pos[r], frontiers[r], env.grid, temp_occupied_cells, env.direction[r]) # plans path for selected drone
                end_path_time = time.time() # sets end time for path planning

                # if valid path was found
                # then add to dynamic obstacles
                if current_path[r] != None:
                    del current_path[r][0] # deletes first step in path since it is the current position of the drone

                    # for debugging
                    if len(current_path[r]):
                        breakpoint

                    # for debugging
                    for path_step, pos in enumerate(current_path[r]):
                        if steps+path_step not in temp_occupied_cells: continue
                        if len(temp_occupied_cells[steps+path_step]) > nr:
                            breakpoint
            
                    # add path to occupied cells
                    # loops through current path of selected drone
                    for path_step, pos in enumerate(current_path[r]):
                        # if the step plus the step of the current path is not in the occupancy state list
                        # then inisialise the step
                        if steps+path_step not in temp_occupied_cells: temp_occupied_cells[steps+path_step] = []
                        # for debugging
                        if pos in temp_occupied_cells[steps+path_step]:
                            breakpoint
                        # for debugging
                        if len(temp_occupied_cells[steps+path_step]) == nr:
                            breakpoint

                        temp_occupied_cells[steps+path_step].append(pos) # add path step to occupany state for current step plus path step

                        # for debugging
                        if len(temp_occupied_cells[steps+path_step]) > nr:
                            breakpoint
                else:
                    breakpoint # for debugging
            
            cntr += 1

            # if all of the paths could be planned
            # then end planning and update occupancy state list with current paths
            # else keep planning
            if cntr == nr and all([current_path[ri] != None for ri in range(nr)]):
                planning = False
                occupied_cells = temp_occupied_cells.copy()
            else:
                planning = True
            
            # if the current path could not be planned
            # then reset priority drone
            # else continue planning for next drone
            if current_path[r] == None:
                prior_r += 1
                # if the priority drone equals the last drone
                # then set to the first drone 
                if prior_r % nr == 0: prior_r = 0
                n_plans += 1
                cntr = 0
                r = prior_r
                temp_occupied_cells = {key: value[:] for key, value in occupied_cells.items()} # reset occupancy state list to original
            else:
                r += 1
            
            # if all drone had te chance to be the priority drone and the planning not complete
            # then check if replanning without on going candidates have been executed
            # if replanning as been executed
            # then the iteration is unsuccessful
            # else replan without on going candidates
            if n_plans == nr:
                # if all of the paths have been replanned including on going paths
                # then the iteration is unsuccessfull
                if replan_ongoing:
                    print("No route")
                    planning_successful = False
                    unsuccessful += 1
                    planning = False
                    for ri in range(nr):
                        env.print_graph(ri, steps-1, trajectory[ri], actions[ri], env.starting_pos[ri], obstacle_layout, dir_path, i, False)
                else:
                    # delete on going path from dynamic obstacles
                    count = list(temp_occupied_cells)[-1]
                    for c in range(steps, count+1):
                        del occupied_cells[c]
                        del temp_occupied_cells[c]

                    current_path = [[] for _ in range(nr)] # reinitialise the current paths

                    # replan all paths again without any ongoing paths
                    replan_ongoing = True
                    prior_r = 0
                    r = prior_r
                    cntr = 0
                    n_plans = 0
        
        # paths were successfully planned
        # then continue to moving the drones
        if planning_successful:
            for r in range(nr):
                # if drone is finished searching
                # then do not move
                if finished_scheduling[r]:
                    continue

                # if candidate is not equal to on going candidate
                # then track scheduled distance of drone
                if frontiers[r] != ongoing_frontiers[r]:
                    sheduled_distances[r].append(distances[env.pos[r].y][env.pos[r].x][frontiers[r].y][frontiers[r].x])

                # add cell to trajectory
                trajectory[r].append(current_path[r][0])

                # if new exploration
                # then track it
                if not env.exploration_grid[current_path[r][0].y, current_path[r][0].x]:
                    explorations[r] += 1
                
                # execute move in environment
                start_dist_time = time.time()
                env.move(r, current_path[r][0])
                end_dist_time = time.time()
                
                # if drone reached frontier
                # AND drone position is equal to starting position
                # then not drone searching state to finished
                # else set on going candidate to current candidate #    <---------------------------------------------------------------------- (why do you check if the drone is at starting position aswell?)
                if env.pos[r] == frontiers[r] and env.pos[r] == env.starting_pos[r]: #    <---------------------------------------------------------------------- (is this check necessary?)
                    if return_home[r] and not finished_scheduling[r]:
                        ongoing_frontiers[r] = frontiers[r]
                        finished_scheduling[r] = True
                        which[r] = 1
                        if not env.exploration_grid.all() and all([finished_scheduling[ri] for ri in range(nr)]):
                            save = True
                        if all([finished_scheduling[ri] for ri in range(nr)]) and np.count_nonzero(env.exploration_grid) != WIDTH*HEIGHT:
                            breakpoint
                    else:
                        ongoing_frontiers[r] = None
                else:
                    ongoing_frontiers[r] = frontiers[r]

                del current_path[r][0] # remove step from current path

                # if drone did not reach end of path
                # OR drone is finished searching
                # then add current candidate to on going candidate
                # else set on going candidate to nothing
                if len(current_path[r]) != 0 or finished_scheduling[r]:
                    ongoing_frontiers[r] = frontiers[r]
                else:
                    ongoing_frontiers[r] = None

                # add move to actions
                if env.prev_pos[r].x < env.pos[r].x: actions[r].append("right")
                if env.prev_pos[r].x > env.pos[r].x: actions[r].append("left")
                if env.prev_pos[r].y > env.pos[r].y: actions[r].append("up")
                if env.prev_pos[r].y < env.pos[r].y: actions[r].append("down")

                # in loop trajectory drawing
                if step_trajectory and i < saved_iterations:
                    if each_drone:
                        if goal_spawning:
                            env.print_graph(r, steps-1, trajectory[r], actions[r], env.starting_pos[r], obstacle_layout, dir_path, i, False, env.goal)
                        else:
                            env.print_graph(r, steps-1, trajectory[r], actions[r], env.starting_pos[r], obstacle_layout, dir_path, i, False)
                    else:
                        if r == nr-1:
                            if goal_spawning:
                                env.print_frame(steps, trajectory, actions, env.starting_pos, obstacle_layout, dir_path, i, True, env.goal)
                            else:
                                env.print_frame(steps, trajectory, actions, env.starting_pos, obstacle_layout, dir_path, i, False, None)

                # exit condition
                # if testing method is set to goal spawning
                # AND any drone has reached the goal
                # then set condition to found goal
                if goal_spawning and np.array([True for j in range(0, nr) if env.pos[j] == env.goal]).any() == True:
                    goal_exit_condition = True
                
                path_time += end_path_time - start_path_time
            
            schedule_times.append(end_scheduler_time-start_scheduler_time)
            path_times.append(path_time)
            dist_update_times.append(end_dist_time-start_dist_time)
            
            # save to draw trajectories
            if i in saves: save = True #    <---------------------------------------------------------------------- (i think old functionality)

    steps_list.append(steps)
    for ri in range(nr):
        explorations_list[ri].append(explorations[ri])

    if save_trajectory and save or i < saved_iterations:
        for ri in range(nr):
            if goal_spawning:
                env.print_graph(ri, steps, trajectory[ri], actions[ri], env.starting_pos[ri], obstacle_layout, dir_path, i, True, env.goal)
            else: 
                env.print_graph(ri, steps, trajectory[ri], actions[ri], env.starting_pos[ri], obstacle_layout, dir_path, i, True, None)

    # calculate flight time and planning time
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

# string of results
print_string = ""
print_string += "\nFOV width: %dm\nFOV height: %dm" %(FOV_W, FOV_H)
print_string += "\nTesting iterations: %d"%(test_iterations)
print_string += "\nTesting time: %.2fh%.2fm%.2fs" %(th,tm,ts)
print_string += "\nAverage planning time: %.2fh%.2fm%.2fs" %(ph,pm,ps)
print_string += "\nAverage steps: %.2f" %(np.mean(np.array(steps_list)))
print_string += "\nAverage flight time: %.2fh%.2fm%.2fs" %(fh,fm,fs)
print_string += "\nAverage flight distance: %.2f m" %(np.mean(np.array(flight_distances)))
for ri in range(nr):
    print_string += "\nAverage explorations for drone %d: %.2f" %(ri, average_explorations[ri])
average_time = np.mean(np.array(schedule_times)) + np.mean(np.array(path_times)) + np.mean(np.array(dist_update_times))
print_string += "\nAverage time scheduling: %.8fs"%(np.mean(np.array(schedule_times)))
print_string += "\nAverage time path planning: %.8fs"%(np.mean(np.array(path_times)))
print_string += "\nAverage time updating distance matrix: %.8fs"%(np.mean(np.array(dist_update_times)))
print_string += "\nAverage time per step: %.8fs"%(average_time)
print_string += "\nPercentage success: %.2f"%((test_iterations-unsuccessful)/test_iterations*100)
print_string += "\nObstacles: %.2f"%(obstacle_density)
for r in range(nr):
    # Count the occurrences of each unique number
    unique_numbers, counts = np.unique(sheduled_distances[r], return_counts=True)
    # Clear the existing figure
    plt.clf()
    # Plotting the histogram
    plt.bar(unique_numbers, counts, color='blue', alpha=0.7)
    plt.xlabel('Planned distances')
    plt.ylabel('Occurrences')
    plt.title('Histogram of distances for drone %d'%(r))
    # Save the histogram as a PNG file
    file_name = 'distance_hist_%d.png' %(r)
    plt.savefig(os.path.join(dir_path, file_name))

print_string += "\nFirst: %.8f, Second: %.8f, Third: %.8f, Fourth: %.8f, Fifth: %.8f, Fifth A: %.8f, Fifth B: %.8f, Fifth B1: %.8f, Fifth B2: %.8f, Sixth: %.8f, Seventh: %.8f"%(np.mean(np.array(env.first)),np.mean(np.array(env.second)),np.mean(np.array(env.third)),np.mean(np.array(env.fourth)),np.mean(np.array(env.fifth)),np.mean(np.array(env.fifth_a)),np.mean(np.array(env.fifth_b)),np.mean(np.array(env.fifth_b_1)),np.mean(np.array(env.fifth_b_2)),np.mean(np.array(env.sixth)),np.mean(np.array(env.seventh)))
print_string += "\nFirst: %.8f, Second: %.8f, Third: %.8f, Fourth: %.8f, Fifth: %.8f"%(np.mean(np.array(env.first_dist)),np.mean(np.array(env.second_dist)),np.mean(np.array(env.thrid_dist)),np.mean(np.array(env.fourth_dist)),np.mean(np.array(env.fifth_dist)))

print(print_string)

# saves results to required location
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
    if goal_spawning:
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

    # env.grid_plot()
    # # self.grid_plot_vec()
    # plt.show()
    # breakpoint
    # plt.close()