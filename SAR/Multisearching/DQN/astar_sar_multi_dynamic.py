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

Point = namedtuple('Point', 'x, y')
HEIGHT = 6
WIDTH = 6

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

class Grid(object):
    def __init__(self, height, width, grid, direction): ########################################### change
        self.grid = grid
        self.width = width
        self.height = height
        self.direction = direction

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

    def neighbors(self, id, path_step):
        (x, y) = id
        
        if preprocessing or not fixed_wing:
            # (right, up, left, down)
            results = [Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        else:
            if self.direction == "right":
                results = [Point(x+1, y), Point(x, y-1), Point(x, y+1)]
            elif self.direction == "left":
                results = [Point(x, y-1), Point(x-1, y), Point(x, y+1)]
            elif self.direction == "up":
                results = [Point(x+1, y), Point(x, y-1), Point(x-1, y)]
            elif self.direction == "down":
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
    def reconstruct_path(self, start, end): #: dict[Location, Location], : Location, : Location
        current = end
        path = [current]
        maneuvers = [False]
        while current != start:
            if current not in self.came_from: print(self.grid, self.start, end)
            new_current = self.came_from[current]
            maneuver = self.came_from_maneuvers[current][1]
            current = new_current
            path.append(current)
            maneuvers.append(maneuver)
        path.reverse()
        maneuvers.reverse()
        return path, maneuvers

    # A* algorithm
    def a_star(self, start, end, grid, dynamic_obstacles, direction=None):
        self.grid = grid
        self.dynamic_obstacles = dynamic_obstacles
        self.graph = Grid(HEIGHT, WIDTH, grid, direction)
        self.start = start
        self.came_from = {}
        self.came_from_maneuvers = {}
        self.cost_so_far = {}
        self.heap = [(0, start, direction)]
        self.cost_so_far[start] = 0
        current = start
        found = False
        
        while len(self.heap) > 0:
            _, current, direction = heapq.heappop(self.heap)
            if current == end:
                found = True
                break
            path, _ = self.reconstruct_path(start, current)
            path_step = len(path)-1
            self.neighbors = self.graph.neighbors(current, path_step)
            for next_node in self.neighbors:
                maneuver = False
                # on location collision
                # if a drone has planned thus far
                # AND a drone is on next_node location
                # AND current is not a maneuver action
                if steps+path_step in self.dynamic_obstacles and next_node in self.dynamic_obstacles[steps+path_step]:
                    # continue
                    maneuver = True
                # cross location collision
                # current is in dynamic obstacles on next step
                if steps+path_step in self.dynamic_obstacles and current in self.dynamic_obstacles[steps+path_step]:
                    # next node is in dynamic obstacles on previous step
                    if steps+path_step-1 in self.dynamic_obstacles and next_node in self.dynamic_obstacles[steps+path_step-1]:
                        # continue
                        maneuver = True
                if maneuver:
                    new_cost = self.cost_so_far[current] + self.heuristic(current, next_node) + 1
                else:
                    new_cost = self.cost_so_far[current] + self.heuristic(current, next_node) #+ self.heuristic(next_node, end)
                if next_node not in self.cost_so_far or new_cost < self.cost_so_far[next_node]: #self.grid[next_node.y, next_node.x] != self.States.OBS.value
                    self.cost_so_far[next_node] = new_cost
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
                    heapq.heappush(self.heap, (priority, next_node, direction))
                    self.came_from[next_node] = current
                    self.came_from_maneuvers[next_node] = (current, maneuver)
        
        if current == end:
            found = True
        # if current != end:
        #     print("bugga%d"%(i))
        if found:
            return self.reconstruct_path(start, end)
        else:
            breakpoint
    
class Environment:
    def __init__(self, nr, obstacles, set_obstacles, obstacle_density):
        # initalise variables
        self.nr = nr
        self.obstacles = obstacles
        self.set_obstacles = set_obstacles
        self.obstacle_density = obstacle_density
        
        # spawn grid
        self.starting_grid = np.zeros((HEIGHT, WIDTH))
        self.grid = self.starting_grid.copy()
        
        # initialise drone
        self.starting_pos = [None]*nr
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()

        # initialise exploration grid
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.starting_grid[y,x] != States.UNEXP.value: self.exploration_grid[y, x] = True

    def reset(self, weight, i, goal_spawning):
        # spawn grid
        global WIDTH
        global HEIGHT
        self.grid = np.zeros((HEIGHT, WIDTH))

        self.weight = weight
        # obstacles
        if self.obstacles:
            if i == 0 or not self.set_obstacles:
                self.starting_grid = np.zeros((int(HEIGHT/2), int(WIDTH/2)), dtype=np.int8)
                
                    # Calculate the number of elements to be filled with 1's
                total_elements = int(HEIGHT/2) * int(WIDTH/2)
                num_ones_to_place = int(self.obstacle_density * total_elements)

                    # Generate random indices to place 1's
                possible_indexes = np.argwhere(np.array(self.starting_grid) == States.UNEXP.value)
                np.random.shuffle(possible_indexes)
                indexes = possible_indexes[:num_ones_to_place]

                    # Set the elements at the random indices to 1
                self.starting_grid[indexes[:, 0], indexes[:, 1]] = States.OBS.value

                self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))

                WIDTH = int(WIDTH/2)*2
                HEIGHT = int(HEIGHT/2)*2

                ES = Enclosed_space_check(int(HEIGHT/2)*2, int(WIDTH/2)*2, self.starting_grid, States)
                self.starting_grid = ES.enclosed_space_handler()
        self.grid = self.starting_grid.copy()
        
        # spawn drone
        indices = np.argwhere(self.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        for ri in range(self.nr):
            self.starting_pos[ri] = Point(indices[0,1], indices[0,0])
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
            indices = np.argwhere(self.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            self.goal = Point(indices[0,1], indices[0,0])

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
        
    def get_max_targets(self, costs):
        # get max value of each dorne
        max_targets_value = [None]*self.nr
        for ri in range(self.nr):
            if not costs[ri]: max_targets_value[ri] = []
            else: max_targets_value[ri] = max(costs[ri].values())

        # get all positions with max value
        max_targets = [[] for i in range(self.nr)]
        for ri in range(self.nr):
            max_targets[ri] = [key for key, value in costs[ri].items() if costs[ri] if value == max_targets_value[ri] ]

        return max_targets
        
    def scheduler(self, ongoing_frontiers):
        # set current positions to explored
        temp_exploration_grid = self.exploration_grid.copy()
        for ri in range(self.nr): 
            temp_exploration_grid[self.pos[ri].y, self.pos[ri].x] = True

        # if no more unexplored cells return to home
        if np.count_nonzero(temp_exploration_grid) == HEIGHT*WIDTH:
            return [self.starting_pos[ri] for ri in range(self.nr)]
        
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
                        costs[ri][Point(x,y)] = 1 / distances[self.pos[ri]][Point(x,y)]

        # set targets based on max costs
        targets = [None]*self.nr
        for ri in range(self.nr):
            if ongoing_frontiers[ri] == None:
                targets[ri] = max(costs[ri], key=costs[ri].get)
            else:
                targets[ri] = ongoing_frontiers[ri]
        
        max_targets = self.get_max_targets(costs)

        # check no targets equal
        # find equal targets
        indices = {}
        for i, item in enumerate(targets):
            if item in indices:
                indices[item].append(i)
            else:
                indices[item] = [i]
        equal_targets = {key: value for key, value in indices.items() if len(value) > 1}

        # if no equal targets
        if not equal_targets:
            return targets

        # delete best targets from temp cost list
        temp_costs = [{key: value for key, value in dictionary.items()} for dictionary in costs]
        for ri in range(self.nr):
            for target in max_targets[ri]:
                del temp_costs[ri][target]

        # check if drones have targets left
        if HEIGHT*WIDTH - np.count_nonzero(temp_exploration_grid) < self.nr:
            # # check if any ongoing frontiers equal to last cell
            # for ri in range(self.nr):
            #     if ongoing_frontiers[ri] != None:
            #         targets[ri] = ongoing_frontiers[ri]
            #         for rj in range(self.nr):
            #             if ongoing_frontiers[rj] == None:
            #                 targets[rj] = self.starting_pos[rj]
            #         return targets
            # get closest drone
            best_drone = None
            cost = 0
            for ri in range(self.nr):
                if cost < costs[ri][targets[ri]]:
                    cost = costs[ri][targets[ri]]
                    best_drone = ri
            for ri in range(self.nr):
                if ri != best_drone: 
                    targets[ri] = self.starting_pos[ri]
                    if self.pos[ri] == self.starting_pos[ri]:
                        breakpoint
            
            return targets

        # find next best targets
        next_max_targets = self.get_max_targets(temp_costs)
        for ri in range(self.nr):
            if ongoing_frontiers[ri] != None:
                max_targets[ri] = [ongoing_frontiers[ri]]
            else:
                max_targets[ri] += next_max_targets[ri]

        # get all combinations of best targets
        combinations = list(product(*max_targets))

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
        for combination in combinations:
            sum_cost = 0
            for i, target in enumerate(combination):
                if target in costs[i]:
                    sum_cost += costs[i][target]
            sum_costs.append(sum_cost)
        
        # set targets to best combination
        max_cost = max(sum_costs)
        best_combination = combinations[sum_costs.index(max_cost)]
        for ri in range(self.nr):
            targets[ri] = best_combination[ri]

        return targets
    
    def cost_function(self, r, occupied_cells):
        # distances = {}
        temp_exploration_grid = self.exploration_grid.copy()

        if len(np.argwhere(temp_exploration_grid == False).tolist()) != 1:
            # could make loop go from current step (save on computations)
            # makes all steps of drone explored
            # for step in occupied_cells:
            #     for ri in range(self.nr):
            #         if occupied_cells[step][ri] != None:
            #             if ri != r:
            #                 temp_exploration_grid[occupied_cells[step][ri].y, occupied_cells[step][ri].x] = True
            
            # makes last step of drone explored
            indices = [None]*self.nr
            for ri in range(self.nr):
                last = False
                if ri == r: continue
                for step, positions in occupied_cells.items():
                    if last: continue
                    if positions[ri] is None:
                        indices[ri] = step-1
                        temp_exploration_grid[occupied_cells[step-1][ri].y, occupied_cells[step-1][ri].x] = True
                        last = True
        else: # if only one cell left to explored return that cell
            index = np.argwhere(temp_exploration_grid == False)[0]
            return Point(index[1], index[0])
        
        # if no more cells left to explore
        # check if selected drone closer to target of other drones
        # if not then return None
        if temp_exploration_grid.all():
            for ri in range(self.nr):
                if ri == r or indices[r] == None: continue
                if distances[env.pos[r]][occupied_cells[indices[r]]] < distances[env.pos[ri]][occupied_cells[indices[ri]]]:
                    return occupied_cells[indices[r]]
            return None
            
        
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
        # if len(labels) == 1:
        #     # Access the nested dictionary for drone position
        #     point_paths = distances[env.pos[r]]
            
        #     # remove points that have been explored
        #     pop_points = []
        #     for point in point_paths:
        #         if temp_exploration_grid[point.y, point.x]:
        #             pop_points.append(point)
        #     for point in pop_points:
        #         point_paths.pop(point)

        #     # # Find the shortest path in the nested dictionary
        #     # min_distance = min(point_paths.values(), key=len)

        #     # Find the key with the shortest path in the nested dictionary
        #     min_point = min(point_paths, key=lambda k: len(point_paths[k]))

        #     if (min_point.x, min_point.y) in labels[0]:
        #         clusters[min_point] = len(labels[0])
        # else:
        for cluster in labels:
            lowest_value = 1000
            for key in cluster:
                if Point(key[0],key[1]) in distances[env.pos[r]] and len(distances[env.pos[r]][Point(key[0],key[1])]) < lowest_value:
                    lowest_key = key
                    lowest_value = len(distances[env.pos[r]][Point(key[0],key[1])])
            clusters[Point(lowest_key[0], lowest_key[1])] = len(cluster)
        
        costs = {}
        for point in clusters:
            in_target = False
            # if distance to point smaller than threshold
            # and point not in any other target cluster
            # add to costs
            # else check for different cluster
            if len(distances[env.pos[r]][point]) < 5:
                for ri in range(self.nr):
                    if ri == r: continue
                    if point in self.target_cluster[ri]:
                        in_target = True
                if not in_target:
                    costs[point] = 1000 / clusters[point]
                    in_target = True
                else:
                    in_target = False
            if not in_target:
                # costs[point] = (1 - distances[point]/(WIDTH + HEIGHT - 1)) + (clusters[point] / (WIDTH*HEIGHT))
                costs[point] = clusters[point] / len(distances[point])
                # costs[point] = (clusters[point]) + self.weight/len(distances[env.pos][point])
                # costs[point] = 1/len(distances[env.pos[r]][point])

        target = max(costs, key=costs.get)
        for cluster in labels:
            if target in cluster:
                self.target_cluster[r] = cluster
                break

        return target
    
    def get_frontier(self, clusters, selected_frontiers, invalid_clusters=None, invalid_frontiers=None):
        final_frontiers = [None]*self.nr
        final_cluster_indices = [None]*self.nr
        final_costs = [None]*self.nr

        for drone in range(self.nr):
            closest_points = {}
            if drone in selected_frontiers: continue
            
            # get closest cells in each cluster
            for cluster_index, cluster in enumerate(clusters):
                if cluster_index in invalid_clusters: continue
                lowest_value = float("inf")
                for key in cluster:
                    if Point(key[0],key[1]) in distances[env.pos[drone]] and len(distances[env.pos[drone]][Point(key[0],key[1])]) < lowest_value and Point(key[0],key[1]) not in invalid_frontiers:
                        lowest_key = key
                        lowest_value = len(distances[env.pos[r]][Point(key[0],key[1])])
                closest_points[Point(lowest_key[0], lowest_key[1])] = [len(cluster), cluster_index]
        
            # get costs of clusters
            costs = {}
            for point in closest_points:
                # if distance to point smaller than threshold
                # and point not in any other target cluster
                # add to costs
                # else check for different cluster
                if len(distances[env.pos[r]][point]) < 2:
                    # costs[point] = 1000 / closest_points[point][0]
                    costs[point] = 1000 / len(distances[env.pos[r]][point])
                else:
                    # costs[point] = (1 - distances[point]/(WIDTH + HEIGHT - 1)) + (clusters[point] / (WIDTH*HEIGHT))
                    costs[point] = closest_points[point][0] / len(distances[env.pos[r]][point])
                    # costs[point] = (clusters[point]) + self.weight/len(distances[env.pos][point])
                    # costs[point] = 1/len(distances[env.pos[r]][point])
            
            final_frontiers[drone] = max(costs, key=costs.get)
            final_cluster_indices[drone] = closest_points[final_frontiers[drone]][1]
            final_costs[drone] = costs[final_frontiers[drone]]
        
        return final_frontiers, final_cluster_indices, final_costs

    def get_clusters(self):
        # divid into different clusters of unexplored regions
        cnter = 0
        labels = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if env.exploration_grid[y,x]:
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
                    if env.exploration_grid[y,x] == env.exploration_grid[y,x+1]:
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
                    if env.exploration_grid[y,x] == env.exploration_grid[y+1,x]:
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

        return labels
    
    def fontier_selector(self, ongoing_frontiers):
        all_selected = False
        invalid_clusters = []
        invalid_frontiers = []
        selected_frontiers = []
        perm_frontiers = ongoing_frontiers.copy()
        clusters = self.get_clusters()
        # select frontiers until all drones have been assigned frontiers
        while not all_selected:
            frontiers, frontier_cluster_index, costs = self.get_frontier(clusters, selected_frontiers, invalid_clusters, invalid_frontiers)
            
            # override selected frontiers
            if len(selected_frontiers) != 0:
                for drone in range(self.nr):
                    if perm_frontiers[drone] != None:
                        frontiers[drone] = perm_frontiers[drone]
                        frontier_cluster_index[drone] = perm_frontiers_cluster_index[drone]
                        costs[drone] = perm_costs[drone]


            # if number of drones smaller or equal to number of clusters
            if self.nr <= len(clusters):
                # check if any frontiers in the same cluster
                same_cluster_drone_indices = []
                same_cluster = False
                for ri in range(self.nr):
                    indices = [i for i, x in enumerate(frontier_cluster_index) if x == frontier_cluster_index[ri]]
                    if len(indices) > 1:
                        same_cluster_drone_indices.append(indices)
                        same_cluster = True
                # if any frontiers in the same cluster
                if same_cluster:
                    # check which drones have better cost,
                    # mark cluster as invalid and
                    # delete other frontiers
                    for cluster_indices in same_cluster_drone_indices:
                        cost = 0
                        for index in cluster_indices:
                            if costs[index] != None and costs[index] > cost:
                                cost = costs[index]
                                best_cost_index = index
                        invalid_clusters.append(frontier_cluster_index[best_cost_index])
                        del cluster_indices[best_cost_index]
                        for index in cluster_indices:
                            frontiers[index] = None
                            frontier_cluster_index[index] = None
                            costs[index] = None
            else:
                # check if any frontiers equal
                same_frontier_drone_indices = []
                same_frontier = False
                for ri in range(self.nr):
                    indices = [i for i, x in enumerate(frontiers) if x == frontiers[ri]]
                    if len(indices) > 1:
                        same_frontier_drone_indices.append(indices)
                        same_frontier = True
                # if any frontiers equal
                if same_frontier:
                    # check which drones have better cost,
                    # mark frontier as invalid and
                    # delete other frontiers
                    for frontier_indices in same_frontier_drone_indices:
                        cost = 0
                        for index in frontier_indices:
                            if costs[index] != None and costs[index] > cost:
                                cost = costs[index]
                                best_cost_index = index
                        invalid_frontiers.append(frontiers[best_cost_index])
                        del frontier_indices[best_cost_index]
                        for index in frontier_indices:
                            frontiers[index] = None
                            frontier_cluster_index[index] = None
                            costs[index] = None

            # check if all selected
            if None not in frontiers:
                all_selected = True
            else:
                # save selected frontiers
                selected_frontiers = []
                for i,frontier in enumerate(frontiers):
                    if frontier != None: selected_frontiers.append(i)
                perm_frontiers = frontiers
                perm_frontiers_cluster_index = frontier_cluster_index
                perm_costs = costs
        
        return frontiers

    
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
                        if actions[i] == "right" and not maneuvers[i]: 
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
        plt.close()

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
in_loop_dist = False
preprocessing = True
fixed_wing = False
test_iterations = 1000
saved_iterations = 0

# environment initialisations
goal_spawning = False
nr = 2
weight = 19
obstacles = True
obstacle_density = 0.6
set_obstacles = True
env = Environment(nr, obstacles, set_obstacles, obstacle_density)

if save_trajectory:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'Astar')      

    date_and_time = datetime.now()
    dir_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(dir_path): os.makedirs(dir_path)

# calculate distances
if not in_loop_dist:
    distances = {}
    print("Preprocessing...")
    starting_time = time.time()
    for dx in range(WIDTH):
        print(dx)
        for dy in range(HEIGHT):
            if env.grid[dy,dx] == States.OBS.value: continue
            temp_paths = {}
            for x in range(WIDTH):
                for y in range(HEIGHT):
                    if env.grid[y,x] != States.OBS.value and (x,y) != (dx,dy):
                        # A* distances
                        # astar = Astar(env.grid)
                        # temp_path = astar.a_star(Point(dx, dy), Point(x,y), env.grid)
                        # del temp_path[0]
                        # temp_paths[Point(x,y)] = temp_path
                        # distances[Point(dx,dy)] = temp_paths

                        # Mannhattan distances
                        distance = env.get_distance(Point(dx, dy), Point(x,y))
                        temp_paths[Point(x,y)] = distance
                        distances[Point(dx,dy)] = temp_paths
    end_time = time.time()
    print("Preprocessing time: %.2fs" %(end_time - starting_time))

    # write to file for later use
    # data_json = convert_data_to_json_format(distances)
    # file_name = "distances.json"
    # file_path = os.path.join(PATH, file_name)
    # write_json(data_json, file_path)
preprocessing = False

# testing loop
testing_start_time = time.time()
steps_list = []
explorations_list = [[] for r in range(nr)]
planning_times = []
flight_times = []
flight_distances = []
schedule_times = []
path_times = []
frontier_list = [[] for _ in range(test_iterations)]
for i in range(test_iterations):
    save = False
    planning_starting_time = time.time()
    if i % 100 == 0: print(i)
    env.reset(weight, i, goal_spawning)
    obstacles = env.exploration_grid.copy()
    steps = 0
    actions = [[] for _ in range(nr)]
    trajectory = [[env.starting_pos[r]] for r in range(nr)]
    maneuvers = [[False] for r in range(nr)]
    current_path = [[] for _ in range(nr)]
    current_maneuvers = [[] for _ in range(nr)]
    done = [False]*nr
    occupied_cells = {}
    occupied_cells[steps] = []
    explorations = [0]*nr
    for r in range(nr):
        occupied_cells[steps].append(env.starting_pos[r])

    ongoing_frontiers = [None]*nr
    exit_condition = False
    while not env.exploration_grid.all() and not exit_condition:
        steps += 1
        if steps not in occupied_cells: occupied_cells[steps] = []

        if env.exploration_grid.all():
            break
        # get frontiers
        if any(frontier is None for frontier in ongoing_frontiers):
            start_scheduler_time = time.time()
            frontiers = env.scheduler(ongoing_frontiers)
            end_scheduler_time = time.time()
        
        frontier_list[i].append(frontiers)

        # plan paths
        path_time = 0
        for r in range(nr):
            if env.pos[r] == frontiers[r] or done[r]:
                if env.pos[r] == env.starting_pos[r]:
                    ongoing_frontiers[r] = frontiers[r]
                    done[r] = True
                continue
            if ongoing_frontiers[r] == None:
                start_path_time = time.time()
                astar = Astar(env.grid)
                current_path[r], current_maneuvers[r] = astar.a_star(env.pos[r], frontiers[r], env.grid, occupied_cells, env.direction[r])
                end_path_time = time.time()
                del current_path[r][0]
                del current_maneuvers[r][0]
            
                # add path to occupied cells
                for path_step, pos in enumerate(current_path[r]):
                    if steps+path_step not in occupied_cells: occupied_cells[steps+path_step] = []
                    occupied_cells[steps+path_step].append(pos)
            
            # execute move 
            trajectory[r].append(current_path[r][0])
            maneuvers[r].append(current_maneuvers[r][0])
            if current_maneuvers[r][0]:
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
                else:
                    ongoing_frontiers[r] = None
            else:
                ongoing_frontiers[r] = frontiers[r]

            del current_path[r][0]
            if len(current_path[r]) != 0:
                ongoing_frontiers[r] = frontiers[r]
            else:
                ongoing_frontiers[r] = None

            # add move to actions
            if env.prev_pos[r].x < env.pos[r].x: actions[r].append("right")
            if env.prev_pos[r].x > env.pos[r].x: actions[r].append("left")
            if env.prev_pos[r].y > env.pos[r].y: actions[r].append("up")
            if env.prev_pos[r].y < env.pos[r].y: actions[r].append("down")

            if in_loop_trajectory and i < saved_iterations:
                if goal_spawning:
                    env.print_graph(r, steps-1, trajectory[r], actions[r], env.starting_pos[r], obstacles, dir_path, i, False, env.goal)
                else:
                    env.print_graph(r, steps-1, trajectory[r], actions[r], env.starting_pos[r], obstacles, dir_path, i, False)

            # exit condition
            if goal_spawning and np.array([True for j in range(0, nr) if env.pos[j] == env.goal]).any() == True:
                exit_condition = True
            
            path_time += end_path_time - start_path_time
        
        schedule_times.append(end_scheduler_time-start_scheduler_time)
        path_times.append(path_time)
        
        if env.pos[0] == env.pos[1] and all([not done[ri] for ri in range(nr)]) and not current_maneuvers[r][0]:
            # save = True
            step = steps
            breakpoint
        if env.prev_pos[0] == env.pos[1] and env.prev_pos[1] == env.pos[0] and all([not done[ri] for ri in range(nr)]) and not current_maneuvers[r][0]:
            # save = True
            step = steps
            breakpoint

    steps_list.append(steps)
    for ri in range(nr):
        explorations_list[ri].append(explorations[ri])

    if save_trajectory and save or i < saved_iterations:
        for ri in range(nr):
            if goal_spawning:
                env.print_graph(ri, steps, trajectory[ri], maneuvers[ri], actions[ri], env.starting_pos[ri], obstacles, dir_path, i, True, env.goal)
            else:
                env.print_graph(ri, steps, trajectory[ri], maneuvers[ri], actions[ri], env.starting_pos[ri], obstacles, dir_path, i, True, None)

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
print_string += "FOV width: %dm\nFOV height: %dm" %(FOV_W, FOV_H)
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