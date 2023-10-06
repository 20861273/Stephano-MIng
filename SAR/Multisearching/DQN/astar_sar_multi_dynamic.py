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

class Grid(object):
    def __init__(self, height, width, grid, direction): ########################################### change
        self.grid = grid
        self.width = width
        self.height = height
        self.direction = direction

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, id, current_id, cr, path_step):
        (x, y) = id
        temp_grid = env.grid.copy()
        if not preprocessing:
            # # if any drones has planned this far ahead
            # # check for collisions with drones
            # if steps+path_step <= len(occupied_cells)-1:
            #     for ri in range(nr):
            #         if cr == ri or occupied_cells[steps+path_step][ri] == None: continue
            #         # on location collision
            #         # make next position of drone an obstacle (steps+path_step=next_position)
            #         temp_grid[occupied_cells[steps+path_step][ri].y, occupied_cells[steps+path_step][ri].x] = States.OBS.value
            #         # cross location collision
            #         # if current location of drone is equal to next position id
            #         # and previous location of drone is equal to current position id
            #         # make current position id obstacle
            #         if occupied_cells[steps+path_step-1][r] ==  Point(x,y)\
            #             and occupied_cells[steps+path_step][r] == Point(current_id[0],current_id[1]):
            #             temp_grid[y, x] = States.OBS.value
            # else:
            #     # trapped collision
            #     # check all possible positions of other drones
            #     # if drone has only one position to move to
            #     # then make it an obstacle for current drone
            #     pass
            for ri in range(nr):
                if r > ri:
                    # on location collision
                    # make next position of drone an obstacle 
                    temp_grid[env.pos[ri].y, env.pos[ri].x] = States.OBS.value

                    # cross location collision
                    # if current location of drone is equal to next position id
                    # and previous location of drone is equal to current position id
                    # make current position id obstacle
                    if env.prev_pos[ri] ==  Point(x,y)\
                        and env.pos[ri] == Point(current_id[0],current_id[1]):
                        temp_grid[y, x] = States.OBS.value

        return temp_grid[y,x] != States.OBS.value

    def is_collision(self, id, cr, cx, cy, path_step):
        (x, y) = id            
        # future collision
        if steps+path_step-1 <= len(occupied_cells)-1:
            # move drones to possible next step locations
            possible_locations = [[] for _ in range(nr)]
            for r in range(nr):
                if r == cr or occupied_cells[steps+path_step-1][r] == None: continue
                if steps+path_step <= len(occupied_cells)-1:
                    if occupied_cells[steps+path_step][r] != None:
                        possible_locations[r] = occupied_cells[steps+path_step][r]
                else:
                    neightbors = [Point(occupied_cells[steps+path_step-1][r].x+1, occupied_cells[steps+path_step-1][r].y),\
                                    Point(occupied_cells[steps+path_step-1][r].x, occupied_cells[steps+path_step-1][r].y-1),\
                                    Point(occupied_cells[steps+path_step-1][r].x-1, occupied_cells[steps+path_step-1][r].y),\
                                    Point(occupied_cells[steps+path_step-1][r].x, occupied_cells[steps+path_step-1][r].y+1)]
                    neightbors = list(filter(self.in_bounds, neightbors))
                    neightbors = list(filter(lambda k: self.is_obstacle(k, id, cr, path_step), neightbors))
                    possible_locations[r] = neightbors
            
            # check if only 1 possible location
            # and if location in new_resutls
            all_valid = True
            for p in possible_locations:
                if len(p) == 1 and p[0] == Point(x,y):
                    all_valid = False
                
            return all_valid

    def neighbors(self, cr, id, step, path_step, occupied_cells):
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
        results = list(filter(lambda k: self.is_obstacle(k, id, cr, path_step), results))
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
        while current != start:
            if current not in self.came_from: print(self.grid, self.start, end)
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path

    # A* algorithm
    def a_star(self, start, end, grid, direction=None, cr=0, step=0, occupied_cells=None):
        self.grid = grid
        self.graph = Grid(HEIGHT, WIDTH, grid, direction)
        self.start = start
        self.came_from = {}
        self.cost_so_far = {}
        self.heap = [(0, start, direction)]
        self.cost_so_far[start] = 0
        current = start
        found = False
        
        while self.heap:
            if current == end:
                found = True
                break
            _, current, direction = heapq.heappop(self.heap)
            path_step = len(self.reconstruct_path(start, current))-1
            self.neighbors = self.graph.neighbors(cr, current, step, len(self.reconstruct_path(start, current))-1, occupied_cells)
            for next_node in self.neighbors:
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
                    heapq.heappush(self.heap, (new_cost, next_node, direction))
                    self.came_from[next_node] = current
        
        if current == end:
            found = True
        # if current != end:
        #     print("bugga%d"%(i))
        if found:
            return self.reconstruct_path(start, end)
        else:
            breakpoint
        # with scheduler not necessary
        # else:
        #     # find lowest cost cell
        #     smallest_non_zero_value = None
        #     smallest_non_zero_positions = []
        #     for position, cost in self.cost_so_far.items():
        #         # Check if the value is non-zero and smaller than the current smallest non-zero value
        #         if cost != 0 and (smallest_non_zero_value is None or cost <= smallest_non_zero_value):
        #             smallest_non_zero_value = cost
        #             smallest_non_zero_positions.append(position)
        #     # find closest cell from lowest cost cells
        #     dist = float('inf')
        #     for position in smallest_non_zero_positions:
        #         if len(distances[position][end]) < dist:
        #             dist = len(distances[position][end])
        #             best_position = position
        #     return self.reconstruct_path(start, best_position)
    
class Environment:
    def __init__(self, nr, obstacle_density):
        # spawn grid
        self.grid = np.zeros((HEIGHT, WIDTH))
        self.nr = nr
        self.obstacle_density = obstacle_density
        
        # spawn drone
        self.starting_pos = [None]*nr
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()
        indices = np.argwhere(self.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        for r in range(self.nr):
            self.starting_pos[r] = Point(indices[0,1], indices[0,0])
            self.pos[r] = self.starting_pos[r]
            self.prev_pos[r] = self.starting_pos[r]
            self.grid[self.starting_pos[r].y, self.starting_pos[r].x] = States.ROBOT.value
            indices_list = indices.tolist()
            del indices_list[0]
            indices = np.array(indices_list)

        # initialise exploration grid
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid[y,x] != States.UNEXP.value: self.exploration_grid[y, x] = True

    def reset(self, weight, i, goal_spawning):
        # spawn grid
        global WIDTH
        global HEIGHT
        self.grid = np.zeros((HEIGHT, WIDTH))

        self.weight = weight
        # obstacles
        if i == 0:
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
        
    def scheduler(self):
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
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    if not temp_exploration_grid[y,x]:
                        if in_loop_dist:
                            costs[ri][Point(x,y)] = 1 / distances[self.pos[ri]][Point(x,y)]
                        else:
                            costs[ri][Point(x,y)] = 1 / len(distances[self.pos[ri]][Point(x,y)])

        # set targets based on costs
        targets = [None]*self.nr
        for ri in range(self.nr):
            targets[ri] = max(costs[ri], key=costs[ri].get)
        
        # check if other drones will reach target first
        # temp_costs = [{key: value for key, value in dictionary.items()} for dictionary in costs]

        # for ri in range(self.nr):
        #     del temp_costs[ri][targets[ri]]

        # for ri in range(self.nr):
        #     closest = False
        #     while not closest:
        #         # if (distance from other drone to other target) + (distance from other target to target) < (distance from drone to target)
        #         # then get next best unexplored cell
        #         # until the truly closest cell is found
        #         for rj in range(self.nr):
        #             if ri == rj: continue
        #             closest = False
        #             if temp_costs or targets[ri] == max(temp_costs[rj], key=costs[rj].get):
        #                 if np.count_nonzero(temp_exploration_grid) == HEIGHT*WIDTH-1:
        #                     closest = True
        #                 else:
        #                     if targets[rj] != targets[ri]:
        #                         if len(distances[self.pos[rj]][targets[rj]]) + len(distances[targets[rj]][targets[ri]]) \
        #                             < len(distances[self.pos[ri]][targets[ri]]):
        #                             del costs[ri][targets[ri]]
        #                             targets[ri] = max(costs[ri], key=costs[ri].get)
        #                         else:
        #                             closest = True
        #                     else:
        #                         if len(distances[self.pos[rj]][targets[rj]]) < len(distances[self.pos[ri]][targets[ri]]):
        #                             del costs[ri][targets[ri]]
        #                             targets[ri] = max(costs[ri], key=costs[ri].get)
        #                         else:
        #                             closest = True
        #             else:
        #                 closest = True

        
        # check no targets equal
        all_selected = False
        while not all_selected:
            # find equal targets
            indices = {}
            for i, item in enumerate(targets):
                if item in indices:
                    indices[item].append(i)
                else:
                    indices[item] = [i]
            equal_targets = {key: value for key, value in indices.items() if len(value) > 1}
            
            # check if any targets were equal
            if equal_targets:
                for target, drones in equal_targets.items():
                    # find best cost for cell
                    max_cost = 0
                    for ri in drones:
                        if costs[ri][target] > max_cost:
                            max_cost = costs[ri][target]
                            best_drone = ri
                    
                    # delete target from other drones costs
                    for ri in drones:
                        if ri == best_drone: continue
                        del costs[ri][target]

                    # get next best target from costs
                    for ri in range(self.nr):
                        if ri == best_drone: continue
                        if not costs[ri]: # if no unexplored cells left return home
                            targets[ri] = self.starting_pos[ri]
                        else:
                            targets[ri] = max(costs[ri], key=costs[ri].get)
            else:
                all_selected = True
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

    
    def print_graph(self, r, steps, path, actions, starting_pos, obstacles, dir_path, cnt, goal_pos=None):
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
                elif Point(j,i) == starting_pos:
                    ax.fill([j + 0.5, j + 1.5, j + 1.5, j + 0.5],\
                            [i + 0.5, i + 0.5, i + 1.5, i + 1.5], \
                                facecolor="red", alpha=0.5)
                elif Point(j,i) == goal_pos:
                    ax.fill([j + 0.5, j + 1.5, j + 1.5, j + 0.5],\
                            [i + 0.5, i + 0.5, i + 1.5, i + 1.5], \
                                facecolor="blue", alpha=0.5)
                    
        # fill explored cells green
        for i, pos in enumerate(path):
            x = pos.x
            y = pos.y
            if i == len(path)-1 and goal_pos == None:
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                    [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                    facecolor="yellow", 
                    alpha=0.5)
            elif goal_pos != Point(x,y):
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
        
        plt_title = "A-star algorithm drone %s: Steps: %s" %(str(r) ,str(steps))
        plt.title(plt_title)

        file_name = "traj%d_%d.png"%(cnt, r)
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
save_trajectory = True
in_loop_dist = True
preprocessing = True
fixed_wing = False

# environment initialisations
goal_spawning = True
nr = 2
weight = 19
obstacle_density = 0
env = Environment(nr, 0)
env.reset(weight, obstacle_density, goal_spawning)

# get distances
distances = {}
if not in_loop_dist:
    print("Preprocessing...")
    starting_time = time.time()
    for dx in range(WIDTH):
        for dy in range(HEIGHT):
            if env.grid[dy,dx] == States.OBS.value: continue
            temp_paths = {}
            for x in range(WIDTH):
                for y in range(HEIGHT):
                    if env.grid[y,x] != States.OBS.value and (x,y) != (dx,dy):
                        astar = Astar(env.grid)
                        temp_path = astar.a_star(Point(dx, dy), Point(x,y), env.grid)
                        del temp_path[0]
                        temp_paths[Point(x,y)] = temp_path
                        distances[Point(dx,dy)] = temp_paths
    end_time = time.time()
    print("Preprocessing time: %.2fs" %(end_time - starting_time))
preprocessing = False

# testing loop
starting_time = time.time()
steps_list = []
average_flight_time = []
average_flight_distance = []
for i in range(5):
    if i % 100 == 0: print(i)
    env.reset(weight,1, goal_spawning)
    obstacles = env.exploration_grid.copy()
    steps = 0
    actions = [[] for _ in range(nr)]
    trajectory = [[env.starting_pos[r]] for r in range(nr)]
    current_path = [[] for _ in range(nr)]
    occupied_cells = {}
    occupied_cells[steps] = []
    closest = [None]*nr
    for r in range(nr):
        occupied_cells[steps].append(env.starting_pos[r])

    ongoing_frontiers = [None]*nr
    exit_condition = False
    while not env.exploration_grid.all() and not exit_condition:
        steps += 1
        if steps not in occupied_cells: occupied_cells[steps] = [None]*nr

        if env.exploration_grid.all():
            break
        # get frontiers
        # frontiers = env.fontier_selector(ongoing_frontiers)
        frontiers = env.scheduler()

        # plan paths
        for r in range(nr):
            if env.pos[r] == frontiers[r]: continue
            astar = Astar(env.grid)
            current_path[r] = astar.a_star(env.pos[r], frontiers[r], env.grid, env.direction[r], r, steps, occupied_cells)
            del current_path[r][0]
            
            # remove future moves from occupied cells
            # count = 0
            # for key, lst in occupied_cells.items():
            #     if lst[r] is not None:
            #         count += 1
            # for c in range(steps, count):
            #     occupied_cells[c][r] = None
            #     if occupied_cells[c][r] == [None]*nr: del occupied_cells[c]
            
            # add current path to occupied cells
            for j, pos in enumerate(current_path[r]):
                if steps+j not in occupied_cells: occupied_cells[steps+j] = [None]*nr
                occupied_cells[steps+j][r] = pos
            
            # execute move
            for ri in range(nr):
                while len(current_path[r]) != 0:
                    trajectory[r].append(current_path[r][0])
                    # occupied_cells[steps].append(current_path[r][0])
                    env.move(r, current_path[r][0])
                    if r == 1:
                        if env.pos[0] == env.pos[1]:
                            breakpoint
                        if env.prev_pos[0] == env.pos[1] and env.prev_pos[1] == env.pos[0]:
                            breakpoint
                    
                    # check if drone reached frontier
                    if env.pos[r] == frontiers[r]:
                        ongoing_frontiers[r] = None
                    else:
                        ongoing_frontiers[r] = frontiers[r]

                    del current_path[r][0]

                    # add move to actions
                    if env.prev_pos[r].x < env.pos[r].x: actions[r].append("right")
                    if env.prev_pos[r].x > env.pos[r].x: actions[r].append("left")
                    if env.prev_pos[r].y > env.pos[r].y: actions[r].append("up")
                    if env.prev_pos[r].y < env.pos[r].y: actions[r].append("down")

                    # exit condition
                    if goal_spawning and np.array([True for j in range(0, nr) if env.pos[j] == env.goal]).any() == True:
                        exit_condition = True

    steps_list.append(steps)
# while not env.exploration_grid.all():
#     steps += 1
#     if steps not in occupied_cells: occupied_cells[steps] = [None]*nr
#     for r in range(nr):
#         count = 0
#         # if reached end of path
#         # if len(current_path[r]) == 0:
#         closest[r] = env.cost_function(r, occupied_cells)
#         if closest[r] == None:
#             continue
#         else:
#             astar = Astar(env.grid)
#             current_path[r] = astar.a_star(env.pos[r], closest[r], env.grid, r, steps, occupied_cells)
#             del current_path[r][0]
        
#         # remove future moves from occupied cells
#         for key, lst in occupied_cells.items():
#             if lst[r] is not None:
#                 count += 1
#         for c in range(steps, count):
#             occupied_cells[c][r] = None
#             if occupied_cells[c][r] == [None]*nr: del occupied_cells[c]
        
#         # add current path to occupied cells
#         for i, pos in enumerate(current_path[r]):
#             if steps+i not in occupied_cells: occupied_cells[steps+i] = [None]*nr
#             occupied_cells[steps+i][r] = pos
        
#         trajectory[r].append(current_path[r][0])
#         # occupied_cells[steps].append(current_path[r][0])
#         env.move(r, current_path[r][0])
#         del current_path[r][0]
#         if env.prev_pos[r].x < env.pos[r].x: actions[r].append("right")
#         if env.prev_pos[r].x > env.pos[r].x: actions[r].append("left")
#         if env.prev_pos[r].y > env.pos[r].y: actions[r].append("up")
#         if env.prev_pos[r].y < env.pos[r].y: actions[r].append("down")

# save_trajectory = False
    if save_trajectory and i < 5:
        if i == 0:
            PATH = os.getcwd()
            PATH = os.path.join(PATH, 'SAR')
            PATH = os.path.join(PATH, 'Results')
            PATH = os.path.join(PATH, 'Astar')      

            date_and_time = datetime.now()
            dir_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
            if not os.path.exists(dir_path): os.makedirs(dir_path)

        for ri in range(nr):
            env.print_graph(ri, steps, trajectory[ri], actions[ri], env.starting_pos[ri], obstacles, dir_path, i, env.goal)

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
    average_flight_time.append(flight_time)
    average_flight_distance.append(flight_distance)

end_time = time.time()

planning_time = end_time - starting_time
pm, ps = divmod(planning_time, 60)
ph = 0
if pm >= 60: ph, pm = divmod(pm, 60)

flight_time = np.mean(np.array(average_flight_time))
fm, fs = divmod(flight_time, 60)
fh = 0
if fm >= 60: fh, fm = divmod(fm, 60)

print("Cell width: %dm\nCell height: %dm" %(FOV_W, FOV_H))
print("Planning time: %.2fh%.2fm%.2fs" %(ph,pm,ps))
print("Average steps: %.2f" %(np.mean(np.array(steps_list))))
print("Average flight time: %.2fh%.2fm%.2fs" %(fh,fm,fs))
print("Average flight distance: %.2f m" %(np.mean(np.array(average_flight_distance))))