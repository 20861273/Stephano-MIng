from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math
import os
from dqn_utils import write_json, read_json

from collections import deque
from itertools import product

from astar_class import Astar
from enclosed_space_checker import Enclosed_space_check

# from sar_dqn_main import COL_REWARD

# Environment characteristics
HEIGHT = 20
WIDTH = 20

# DENSITY = 30 # percentage

# Direction states
class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    STAY = 4

# Block states
class States(Enum):
    UNEXP = 0
    OBS = 1
    ROBOT = 4
    GOAL = 2
    EXP = 3

# Setup position variable of robot as point
Point = namedtuple('Point', 'x, y')


class Environment:
    """
    A class used to represent the environment

    ...

    Attributes
    ----------
    nr : int
        number of drones
    pos : list of named tuples
        current positions of drones
    prev_pos : list of named tuples
        previous positions of drones
    starting_pos : list of named tuples
        starting positions of drones
    positive_reward : float
        positive reward for a successful episode
    negative reward : float
        negative reward for termination state
    positive_exploration_reward : flaot
        positive reward for exploring new cells
    negative_step_reward : float
        negative reward for taking step
    collision : dictionary
        keeps track of collision states for each drone
    encoding : string
        keeps track of input encoding used
    lidar : boolean
        enables or disables LiDAR of surrounding cells
    obstacles : boolean
        enables or disables obstacles

    Methods
    -------
    generate_grid()
        generates grid
    reset()
        resets environment
    get_state()
        returns state of agents
    step_centralized(actions)
        takes step and returns observations
    calc_reward_centralized()
        calculates reward for current step
    get_distance(end, start)
        returns manhattan distance between end and start
    get_direction(goal, r_i)
        returns direction from drone r_i to goal cell
    get_closest_unexplored(r_i, other_location)
        returns distance from closest unexplored cell to drone r_i
    is_collision_centralized(pt, r_i, x ,y, action)
        checks if any new positions (pt) collides with new position (x,y) of current drone (r_i) when taking action (action)
    _update_env()
        updates environment
    _move_centralized(action)
        moves all drones
    check_surrounding_cells(r_i)
        checks surrounding cells of drone r_i
    
    """
    
    def __init__(self, nr, obstacles, set_obstacles, obstacle_density, obstacle_path,\
                reward_system, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, \
                training_type, encoding, guide, lidar, refuel, curriculum_learning, total_episodes, total_steps, exp, ts, save_path, goal_spawning=False):
        
        self.nr = nr
        self.cntr = 0

        # Set robot(s) position
        self.pos = [Point(0,0)]*self.nr
        self.prev_pos = [Point(0,0)]*self.nr
        self.starting_pos = [Point(0,0)]*self.nr

        self.refuel = refuel

        if self.refuel: self.starting_fuel = (WIDTH+HEIGHT)*2
        else: self.starting_fuel = HEIGHT*WIDTH*3
        self.fuel = self.starting_fuel
        self.guide = guide

        self.curriculum_learning = curriculum_learning
        self.total_episodes = total_episodes
        self.stage = 1
        
        # Generates grid (Grid[y,x])
        self.obstacles = obstacles
        self.obstacle_end_density = obstacle_density
        self.obstacle_density = obstacle_density
        self.obstacle_path = obstacle_path
        
        if self.obstacles:
            if not set_obstacles:
            # if exp == 0 and ts == 0 and not set_obstacles:
                self.generate_grid()
                self.starting_grid = self.grid.copy()
                # self.starting_grid = self.generate_maze(HEIGHT, WIDTH)
                # self.draw_edges()
                self.set_obstacles()

                self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))
                
                ES = Enclosed_space_check(HEIGHT, WIDTH, self.starting_grid, States)
                self.starting_grid = ES.enclosed_space_handler()
                self.grid = self.starting_grid.copy()

                file_name = "grid.json"
                file_name = os.path.join(save_path, file_name)
                write_json(self.grid.tolist(), file_name)
            elif set_obstacles:
                file_name = "grid.json"
                file_name = os.path.join(self.obstacle_path, file_name)
                self.starting_grid = np.array(read_json(file_name))
                self.grid = self.starting_grid
            # self.generate_grid()
            # self.starting_grid = self.grid.copy()
            # self.set_obstacles()

            # self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))
            
            # ES = Enclosed_space_check(HEIGHT, WIDTH, self.starting_grid, States)
            # self.starting_grid = ES.enclosed_space_handler()
            # self.grid = self.starting_grid.copy()
            else:
                file_name = "grid.json"
                file_name = os.path.join(save_path, file_name)
                self.starting_grid = np.array(read_json(file_name))
                self.grid = self.starting_grid
        else:
            self.generate_grid()
            self.starting_grid = self.grid.copy()
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        self.goal_state = np.ones((HEIGHT, WIDTH), dtype=np.bool_)

        # Set robot(s) position
        self.pos = self.starting_pos

        for i in range(self.nr):
            self.exploration_grid[self.pos[i].y, self.pos[i].x] = True
        explored = np.argwhere(self.grid == States.OBS.value)
        for cell in explored:
            self.exploration_grid[cell[0], cell[1]] = True

        self.reward_system = reward_system
        self.positive_reward = positive_reward
        self.positive_exploration_reward = positive_exploration_reward
        self.negative_reward = negative_reward
        self.negative_step_reward = negative_step_reward

        self.unexplored = False
        self.collision = {  'obstacle' :   [False]*self.nr,
                            'boundary' :   [False]*self.nr,
                            'drone'    :   [False]*self.nr,
                            'trapped'  :   [False]*self.nr}
        self.collision_state = [False]*self.nr

        self.done = [False]*self.nr

        self.training_type = training_type
        self.encoding = encoding
        self.lidar = lidar
        self.return_home = False
        self.return_home_cell = [None]*self.nr
        self.astar = Astar(HEIGHT, WIDTH, self.grid, States)

        self.goal_spawning = goal_spawning

        # print("\nGrid size: ", self.grid.shape)

    def generate_grid(self):
        # Generate grid of zeros
        # For no obstacles: generates grid of H*W
        # For obstacles: generates grid of (H/2 * W/2) since maze corridors should be 2 wide (grid will be scaled to H*W later in code)
        if self.obstacles: self.grid = np.zeros((int(HEIGHT/2), int(WIDTH/2)), dtype=np.int8)
        else: self.grid = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        # self.draw_edges()

        # for curriculum learning: steadily increases size of environment by making border size smaller
        if self.curriculum_learning['collisions']:
            self.draw_bounds(0)

    def reset(self, current_episode, percentage = 0.0):
        self.collision = {  'obstacle' :   [False]*self.nr,
                            'boundary' :   [False]*self.nr,
                            'drone'    :   [False]*self.nr,
                            'trapped'  :   [False]*self.nr}
        self.collision_state = [False]*self.nr
        # Clear all visited blocks
        if self.obstacles: # and current_episode/self.total_episodes > 0.5:

            self.generate_grid()
            self.starting_grid = self.grid.copy()
            self.set_obstacles()

            self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))
                
            ES = Enclosed_space_check(HEIGHT, WIDTH, self.starting_grid, States)
            self.starting_grid = ES.enclosed_space_handler()
            self.grid = self.starting_grid.copy()

            # self.grid = self.starting_grid.copy()

            # random grid
            # self.generate_grid()
            # self.starting_grid = self.grid.copy()
            # self.set_obstacles()

            # self.starting_grid = np.kron(self.starting_grid, np.ones((2, 2)))
            
            # ES = Enclosed_space_check(HEIGHT, WIDTH, self.starting_grid, States)
            # self.starting_grid = ES.enclosed_space_handler()
            # self.grid = self.starting_grid.copy()
        else:
            self.grid.fill(0)
        # self.grid = self.generate_map()
        # self.grid = self.generate_maze(HEIGHT, WIDTH)
        # self.draw_edges()
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)

        self.calculate_distances()
        
        if self.curriculum_learning['collisions']:
            self.draw_bounds(current_episode, percentage)

        explored = np.argwhere(self.grid == States.OBS.value)
        for cell in explored:
            self.exploration_grid[cell[0], cell[1]] = True

        self.fuel = self.starting_fuel

        # Static goal location spawning
        # self.goal = Point(0,0)
        # grid[self.goal.y, self.goal.x] = States.GOAL.value

        # Random goal location spawning
        if self.curriculum_learning['sparse reward']:
            self.clear_rows = np.where(np.any(self.grid == States.UNEXP.value, axis=1))[0]
            if self.clear_rows.shape[0]:
                np.random.shuffle(self.clear_rows)
                self.clear_rows = self.clear_rows[:5]
            self.goal = [0]*self.clear_rows.shape[0]
            for i, row in enumerate(self.clear_rows):
                indices = np.argwhere(self.grid == States.UNEXP.value)
                np.random.shuffle(indices)
                self.goal[i] = Point(indices[0,1], row)
                self.grid[self.goal[i].y, self.goal[i].x] = States.GOAL.value
        else:
            indices = np.argwhere(self.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            self.goal = Point(indices[0,1], indices[0,0])
            self.grid[self.goal.y, self.goal.x] = States.GOAL.value

        # Setup agent
        # Set new starting pos
        if self.training_type == "decentralized":
            border_spawning = False
            if border_spawning:
                for i in range(0, self.nr):
                    indices = np.argwhere(self.grid == States.UNEXP.value)
                    indices = list(filter(self._is_boundary, indices))
                    np.random.shuffle(indices)
                    self.starting_pos[i] = Point(indices[i,1], indices[i,0])
                    self.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
                
            else:
                for i in range(0, self.nr):
                    indices = np.argwhere(self.grid == States.UNEXP.value)
                    np.random.shuffle(indices)
                    self.starting_pos[i] = Point(indices[i,1], indices[i,0])
                    self.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
        elif self.training_type == "centralized":
            for i in range(0, self.nr):
                indices = np.argwhere(self.grid == States.UNEXP.value)
                np.random.shuffle(indices)
                self.starting_pos[i] = Point(indices[0,1], indices[0,0])
                # checks if drones are spawned next to each other
                # if i > 1:
                #     remove_indices = []
                #     for r_i in range(self.nr):
                #         for j in range(len(indices)):
                #             if self.get_distance(Point(indices[j,1], indices[j,0]), self.starting_pos[r_i]) < 2 and j not in remove_indices:
                #                 remove_indices.append(j)

                #     indices = [element for index, element in enumerate(indices) if index not in remove_indices]
                #     indices = np.array(indices)
                                            
                # self.starting_pos[i] = Point(indices[0,1], indices[0,0])
                self.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()        

        for i in range(self.nr):
            self.exploration_grid[self.pos[i].y, self.pos[i].x] = True

        self.direction = [(Direction.RIGHT).value for i in range(self.nr)]
                
        self.score = 0

        self.astar = Astar(HEIGHT, WIDTH, self.grid, States)

        # set target cluster
        self.target_cluster = [[None]]*self.nr

        self.done = [False]*self.nr
        
        image_state, non_image_state, closest_unexplored = self.get_state()

        self.prev_closest_unexplored = closest_unexplored.copy()

        return image_state, non_image_state

    def step_centralized(self, actions):
        self.collision = {  'obstacle' :   [False]*self.nr,
                            'boundary' :   [False]*self.nr,
                            'drone'    :   [False]*self.nr,
                            'trapped'  :   [False]*self.nr}
        self.collision_state = [False]*self.nr
        self.cntr += 1
        self.score = 0

        # 2. Do action
        self._move_centralized(actions) # update the robot
            
        # 3. Update score and get state
        game_over = False           

        # 4. Update environment
        self._update_env()

        image_state, non_image_state, closest_unexplored = self.get_state()
        # self.ongoing_frontier = closest_unexplored.copy()

        self.score += self.calc_reward_centralized(closest_unexplored)
        self.score -= self.negative_step_reward

        # checks if no actions
        for i in range(self.nr):
            if actions[i] == Direction.STAY.value:
                self.score -= self.negative_reward
                game_over = True  
                reward = self.score
                return image_state, non_image_state, reward, game_over, (0, self.collision)

        # refuel at home
        if np.array([True for i in range(0, self.nr) if self.pos[i] == self.starting_pos[i]]).any() == True:
            self.fuel = self.starting_fuel

        # 5. Check exit condition
        if any(self.collision_state):
            self.score -= self.negative_reward
            reward = self.score
            game_over = True
            return image_state, non_image_state, reward, game_over, (0, self.collision)
        
        if self.fuel == 0:
            self.score -= self.negative_reward
            reward = self.score
            game_over = True
            return image_state, non_image_state, reward, game_over, (0, self.collision)

        if self.reward_system["find goal"] or self.goal_spawning:
            if self.curriculum_learning['sparse reward']:
                # if self._found_all_goals():
                #     self.score += self.positive_reward
                #     reward = self.score
                #     game_over = True
                #     return image_state, non_image_state, reward, game_over, 1
                if self._found_a_goal():
                    if len(self.goal) == 0:
                        self.score += self.positive_reward
                        reward = self.score
                        game_over = True
                        return image_state, non_image_state, reward, game_over, (1, self.collision)
                    self.score += self.positive_exploration_reward
            else:
                if np.array([True for i in range(0, self.nr) if self.pos[i] == self.goal]).any() == True:
                    self.score += self.positive_reward
                    reward = self.score
                    game_over = True
                    return image_state, non_image_state, reward, game_over, (1, self.collision)
        elif self.reward_system["coverage"]:
            if self.exploration_grid.all():
                self.score += self.positive_reward
                reward = self.score
                game_over = True
                return image_state, non_image_state, reward, game_over, (1, self.collision)
        
        self.fuel -= 1
        # 6. return game over and score
        reward = self.score
        return image_state, non_image_state, reward, game_over, (0, self.collision)
    
    def step_decentralized(self, actions):
        # action = self.decode_action(action)
        
        self.score = [0]*self.nr

        # 2. Do action
        self._move_decentralized(actions) # update the robot
            
        # 3. Update score and get state
        game_over = False           

        # 4. Update environment
        self._update_env()

        state = self.get_state()

        temp_score = [agnet_score + reward for agnet_score, reward in zip(self.score, self.calc_reward_decentralized())]
        self.score = temp_score
        temp_score = [agnet_score - self.negative_step_reward for agnet_score in self.score]
        self.score = temp_score
        reward = self.score

        # 5. Check exit condition
        if self.collision_state:
            reward = self.score
            game_over = True
            return state, reward, game_over, None

        found_goal = [i for i in range(0, self.nr) if self.pos[i] == self.goal]
        if len(found_goal) != 0:
        # if (self.exploration_grid == self.goal_state).all():
            self.score[found_goal[0]] += self.positive_reward

            # Give other agents half positive reward
            # for i in range(self.nr):
            #     if found_goal[0] == i: self.score[i] += self.positive_reward
            #     else: self.score[i] += self.positive_reward/2

            reward = self.score
            game_over = True
            for i in range(0, self.nr):
                if self.pos[i] == self.goal:
                    return state, reward, game_over, i

        # 6. return game over and score
        return state, reward, game_over, None
    
    def decode_action(self, action):
        temp_action = action
        actions_completed = [0] * self.nr
        for i in range(self.nr):
            digit = temp_action % 4
            person = self.nr - i - 1
            actions_completed[person] = digit
            temp_action //= 4
        return actions_completed

    # give positive reward for each new cell explored
    def calc_reward_centralized(self, closest_unexplored):
        temp_arr = np.array([False]*self.nr)
        score = 0
        # for i in range(0, self.nr):
        #     if self.exploration_grid[self.pos[i].y, self.pos[i].x] == False:
        #         temp_arr[i] = True
        #     if temp_arr[i] == True:
        #         score += self.positive_exploration_reward
        #         # score += np.count_nonzero(self.exploration_grid) / self.exploration_grid.size
        #     elif self.return_home and self.pos[i] == self.return_home_cell[i]:
        #         score += self.positive_exploration_reward

        for i in range(0, self.nr):
            if self.pos[i] == self.prev_closest_unexplored[i]: 
                score += self.positive_exploration_reward
                if self.pos[i] == self.starting_pos[i]:
                    self.done[i] = True

        self.prev_closest_unexplored = closest_unexplored.copy()
        return score
    
    def calc_reward_decentralized(self):
        temp_arr = np.array([False]*self.nr)
        score = [0]*self.nr
        for i in range(0, self.nr):
            if self.exploration_grid[self.pos[i].y, self.pos[i].x] == False:
                temp_arr[i] = True
            if temp_arr[i] == True:
                score[i] += self.positive_exploration_reward
                # score[i] += np.count_nonzero(self.exploration_grid) / self.exploration_grid.size
        return score
    
    def get_distance(self, end, start):
        return abs(start.x - end.x) + abs(start.y - end.y)
    
    def get_direction(self, goal, r_i):
        if goal.x > self.pos[r_i].x:
            # direction = Direction.RIGHT.value
            direction = [1,0,0,0]
            # self.return_home_cell[r_i] = Point(self.pos[r_i].x+1, self.pos[r_i].y)
        elif goal.x < self.pos[r_i].x:
            # direction = Direction.LEFT.value
            direction = [0,1,0,0]
            # self.return_home_cell[r_i] = Point(self.pos[r_i].x-1, self.pos[r_i].y)
        elif goal.y > self.pos[r_i].y:
            # direction = Direction.DOWN.value
            direction = [0,0,1,0]
            # self.return_home_cell[r_i] = Point(self.pos[r_i].x, self.pos[r_i].y+1)
        elif goal.y < self.pos[r_i].y:
            # direction = Direction.UP.value
            direction = [0,0,0,1]
            # self.return_home_cell[r_i] = Point(self.pos[r_i].x, self.pos[r_i].y-1)
        else:
            direction = [0,0,0,0]

        return direction
    
    def get_closest_unexplored(self, r_i, other_location):
        distances = {}
        temp_exploration_grid = self.exploration_grid.copy()
        for i in range(self.nr): 
            temp_exploration_grid[self.pos[i].y, self.pos[i].x] = True

        if np.count_nonzero(temp_exploration_grid == False) == 1: other_location = None
        
        # gets the distance to all unvisited blocks
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if temp_exploration_grid[y,x] == False:
                    distance = self.get_distance(Point(x,y), self.pos[r_i])

                    if not other_location == Point(x,y):
                        distances[Point(x,y)] = distance
        
        # checks if cell reachable
        if not distances:
            return None
        else:
            return min(distances, key=distances.get) # returns position
            # return distances[min(distances, key=distances.get)] #returns distance

    def calculate_distances(self):
        num_cells = HEIGHT * WIDTH
        self.distances = np.full((num_cells, num_cells), HEIGHT * WIDTH)  # Initialize distances to HEIGHT*WIDTH (unreachable)

        # Define movements (up, down, left, right)
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for start_cell in range(num_cells):
            if self.starting_grid[start_cell // HEIGHT, start_cell % HEIGHT] == States.UNEXP.value:
                queue = deque([(start_cell, 0)])  # Initialize queue with the starting cell and distance 0
                visited = np.zeros(num_cells, dtype=bool)  # Mark all cells as not visited
                visited[start_cell] = True  # Mark the starting cell as visited

                while queue:
                    current_cell, distance = queue.popleft()
                    self.distances[start_cell, current_cell] = distance

                    # Explore neighbors
                    row, col = current_cell // HEIGHT, current_cell % HEIGHT
                    for move_row, move_col in moves:
                        new_row, new_col = row + move_row, col + move_col
                        neighbor_cell = new_row * HEIGHT + new_col

                        if 0 <= new_row < WIDTH and 0 <= new_col < HEIGHT \
                            and not visited[neighbor_cell] and self.starting_grid[new_row, new_col] == States.UNEXP.value:
                            queue.append((neighbor_cell, distance + 1))
                            visited[neighbor_cell] = True
    
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
    
    def scheduler(self):
        # set current positions to explored
        temp_exploration_grid = self.exploration_grid.copy()
        for ri in range(self.nr): 
            temp_exploration_grid[self.pos[ri].y, self.pos[ri].x] = True

        # if no more unexplored cells return to home
        if np.count_nonzero(temp_exploration_grid) == HEIGHT*WIDTH:
            return [self.starting_pos[ri] for ri in range(self.nr)]

        # get neighbours
        neighbours = [[] for _ in range(self.nr)]
        
        # get costs of unexplored cells
        costs = [{} for _ in range(self.nr)]
        for ri in range(self.nr):
            # if ongoing_frontiers[ri] != None: continue
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    if not temp_exploration_grid[y,x] and True not in [Point(x,y) in neighbours[ri] for ri in range(self.nr)]:
                        # costs[ri][Point(x,y)] = distances[self.pos[ri]][Point(x,y)]
                        costs[ri][Point(x,y)] = self.distances[self.pos[ri].y*HEIGHT + self.pos[ri].x][y*HEIGHT + x]

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
                min_targets[ri] += next_min_targets[ri]

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
            min_targets[ri].append(self.starting_pos[ri])

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
                if target == self.starting_pos[ri]:
                    sum_cost += penalty
            sum_costs.append(sum_cost)
        
        # set targets to best combination
        min_cost = min(sum_costs)
        best_combination = combinations[sum_costs.index(min_cost)]
        for ri in range(self.nr):
            targets[ri] = best_combination[ri]
            if self.pos[ri] != self.starting_pos[ri]:
                self.done[ri] = False
            if targets[ri] != self.starting_pos[ri]:
                self.done[ri] = False

        if not self.exploration_grid.all() and all([targets[ri] == self.starting_pos[ri] for ri in range(self.nr)]) and targets[0] == self.starting_pos[0]:
            breakpoint

        return targets

    # def scheduler(self):
    #     # set current positions to explored
    #     temp_exploration_grid = self.exploration_grid.copy()
    #     for r in range(self.nr): 
    #         temp_exploration_grid[self.pos[r].y, self.pos[r].x] = True

    #     # if no more unexplored cells return to home
    #     if np.count_nonzero(temp_exploration_grid) == HEIGHT*WIDTH:
    #         return [self.starting_pos[r] for r in range(self.nr)]

    #     # gets the distance to all unvisited blocks
    #     distances = [{} for _ in range(self.nr)]
    #     for r in range(self.nr):
    #         for y in range(self.grid.shape[0]):
    #             for x in range(self.grid.shape[1]):
    #                 if not temp_exploration_grid[y,x]:
    #                     distance = self.get_distance(Point(x,y), self.pos[r])
    #                     distances[r][Point(x,y)] = distance

    #     # get costs of unexplored cells
    #     costs = [{} for _ in range(self.nr)]
    #     for r in range(self.nr):
    #         for y in range(self.grid.shape[0]):
    #             for x in range(self.grid.shape[1]):
    #                 if not temp_exploration_grid[y,x]:
    #                     costs[r][Point(x,y)] = 1 / distances[r][Point(x,y)]

    #     # set targets based on costs
    #     targets = [None]*self.nr
    #     for r in range(self.nr):
    #         targets[r] = max(costs[r], key=costs[r].get)
        
    #     # check no targets equal
    #     all_selected = False
    #     while not all_selected:
    #         # find equal targets
    #         indices = {}
    #         for i, item in enumerate(targets):
    #             if item in indices:
    #                 indices[item].append(i)
    #             else:
    #                 indices[item] = [i]
    #         equal_targets = {key: value for key, value in indices.items() if len(value) > 1}
            
    #         # check if any targets were equal
    #         # equal and more than unexplored cells than number of drones 
    #         if equal_targets: 
    #             for target, drones in equal_targets.items():
    #                 # find best cost for cell
    #                 max_cost = 0
    #                 for r in drones:
    #                     if costs[r][target] > max_cost:
    #                         max_cost = costs[r][target]
    #                         best_drone = r
                    
    #                 # delete target from other drones costs
    #                 for r in drones:
    #                     if r == best_drone: continue
    #                     del costs[r][target]

    #                 # check if no cells left
    #                 if not costs[r]:
    #                     targets[r] = self.starting_pos[r]
    #                     all_selected = True
    #                     break

    #                 # get next best target from costs
    #                 for r in range(self.nr):
    #                     if r == best_drone: continue
    #                     if not costs[r]: # if no unexplored cells left return home
    #                         targets[r] = self.starting_pos[r]
    #                     else:
    #                         targets[r] = max(costs[r], key=costs[r].get)
    #         else:
    #             all_selected = True
    #     return targets

    def cost_function(self, r_i, other_location):
        distances = {}
        temp_exploration_grid = self.exploration_grid.copy()
        for i in range(self.nr): 
            temp_exploration_grid[self.pos[i].y, self.pos[i].x] = True

        # if only one unexplored cell left
        if np.count_nonzero(temp_exploration_grid == False) == 1: other_location = None
        elif other_location != None: temp_exploration_grid[other_location.y, other_location.x] = True
        
        # gets the distance to all unvisited blocks
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if not temp_exploration_grid[y,x]:
                    distance = self.get_distance(Point(x,y), self.pos[r_i])
                    distances[Point(x,y)] = distance

        if not distances:
            return None
        # if min(distances, key=distances.get) == 1:
        #     return min(distances, key=distances.get)

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
        
        clusters = {}
        if len(labels) == 1:
            min_distance = min(distances, key=distances.get)
            if min_distance in labels[0]:
                clusters[min_distance] = len(labels[0])
        else:
            for cluster in labels:
                lowest_value = 1000
                selected = False
                for key in cluster:
                    if key in distances and distances[key] < lowest_value:
                        lowest_key = key
                        lowest_value = distances[key]
                    clusters[Point(lowest_key[0], lowest_key[1])] = len(cluster)

        # costs = {}
        # for point in clusters:
        #     in_target = False
        #     if distances[point] < 5:
        #         for r in range(self.nr):
        #             if r == r_i: continue
        #             if point in self.target_cluster[r]:
        #                 in_target = True
        #         if not in_target:
        #             costs[point] = 1000 / clusters[point]
        #             in_target = True
        #         else:
        #             in_target = False
        #     if not in_target:
        #         # costs[point] = (1 - distances[point]/(WIDTH + HEIGHT - 1)) + (clusters[point] / (WIDTH*HEIGHT))
        #         costs[point] = clusters[point] / distances[point]
        #         # costs[point] = (clusters[point]) + 10/(distances[point])
        
        costs = {}
        for point in clusters:
            costs[point] = 1 / distances[point]
        
        # make sure no other drones are targeting the same target
        # check = False
        # while check:
        #     target = max(costs, key=costs.get)
        #     for r in range(self.nr):
        #         if r == r_i: continue
        #         if target == self.target[r]:
        #             del costs[target]
        #         else:
        #             check = True
        #             self.target[r_i] = target

        # make sure no other drones are targeting the same cluster
        target = max(costs, key=costs.get)
        for cluster in labels:
            if target == cluster:
                self.target_cluster[r_i] = cluster
                break

        return target

    def get_state(self, x1=None, y1=None, x2=None, y2=None):
        non_image_state = [None]*self.nr
        if self.guide:
            closest_unexplored = [None]*self.nr
            closest_unexplored_dist = [None]*self.nr
            for r_i in range(self.nr):
                # checks if enough fuel to get home
                if self.refuel and self.fuel == self.get_distance(self.pos[r_i], self.starting_pos[r_i]) and self.pos[r_i]!= self.starting_pos[r_i]:
                    closest_unexplored[r_i] = self.starting_pos[r_i]
                    self.return_home = True
                else:
                    if r_i == 0: other_location = None
                    else: other_location = closest_unexplored[r_i-1]
                    closest_unexplored[r_i] = self.get_closest_unexplored(r_i, other_location)
                    self.return_home = False

            for r_i in range(self.nr):
                if closest_unexplored[r_i] == None:
                    if self.refuel:
                        # non_image_state = [[self.fuel, 0]]
                        non_image_state[r_i] = [self.fuel, 0, 4]
                        # non_image_state = [[self.fuel, 0, 0,0,0,0]]
                    else:
                        # non_image_state[r_i] = [WIDTH*HEIGHT]
                        closest_unexp_map = np.zeros(self.grid.shape)
                        # non_image_state[r_i] = [0]
                        # non_image_state[r_i] = [4]
                        # non_image_state[r_i] = [0,0,0,0]
                        # non_image_state[r_i] = [0, 4]
                        # non_image_state[r_i] = [0, 0,0,0,0]
                        if self.lidar:
                            # non_image_state[r_i] = [0] + self.check_surrounding_cells(r_i)
                            # non_image_state[r_i] = [4] + self.check_surrounding_cells(r_i)
                            non_image_state[r_i] = [0, 4] + self.check_surrounding_cells(r_i)
                else:
                    # print(closest_unexplored, closest_unexplored[r_i])
                    # path = self.astar.a_star(self.pos[r_i], closest_unexplored[r_i])
                    # del path[0]

                    # closest_unexplored_dist[r_i] = self.get_distance(self.pos[r_i], closest_unexplored[r_i])

                    if self.refuel:
                        # non_image_state = [self.fuel, len(path)]
                        # non_image_state[r_i] = [self.fuel, len(path), self.get_direction(path[0], r_i)]
                        non_image_state[r_i] = [self.fuel, closest_unexplored_dist[r_i], self.get_direction(closest_unexplored[r_i], r_i)]
                        # non_image_state = [[self.fuel, len(path)] + self.get_direction(path[0])]
                        # non_image_state = [self.fuel, closest_unexplored]
                    else:
                        non_image_state[r_i] = [closest_unexplored[r_i].x*WIDTH + closest_unexplored[r_i].y] # cell test
                        
                        # non_image_state[r_i] = [closest_unexplored] 
                        # non_image_state[r_i] = [len(path)] # Test 1
                        # non_image_state[r_i] = [self.get_direction(path[0], r_i)] # Test 3
                        # non_image_state[r_i] = self.get_direction(path[0], r_i) # Test 7
                        # non_image_state[r_i] = [len(path), self.get_direction(path[0], r_i)] # Test 2: a star distance (with obstacles)
                        # non_image_state[r_i] = [closest_unexplored_dist[r_i], self.get_direction(closest_unexplored[r_i], r_i)] # Test 2: manhattan distance (without obstacles)
                        # non_image_state[r_i] = [len(path)] + self.get_direction(path[0], r_i)
                        if self.lidar:
                            # non_image_state[r_i] = [len(path)]  + self.check_surrounding_cells(r_i)
                            # non_image_state[r_i] = [self.get_direction(path[0], r_i)]  + self.check_surrounding_cells(r_i)
                            # non_image_state[r_i] = [len(path), self.get_direction(path[0], r_i)]  + self.check_surrounding_cells(r_i)
                            non_image_state[r_i] = [closest_unexplored_dist[r_i], self.get_direction(closest_unexplored[r_i], r_i)]  + self.check_surrounding_cells(r_i) # Test 2: manhattan distance (without obstacles)
                    
        # get distances of drones
        # for r_c in range(self.nr):
        #     distances = []
        #     for r_i in range(self.nr):
        #         if r_c == r_i: continue
        #         distances.append(self.get_distance(self.pos[r_c], self.pos[r_i]))
        #     non_image_state[r_c] = non_image_state[r_c] + distances
        
        # Position state: flattened array of environment where position is equal to 1
        if self.encoding == "position" or self.encoding == "position_exploration" or self.encoding == "position_occupancy":
            if self.lidar:
                # percentage_explored = [np.mean(self.exploration_grid)]
                surroundings_state = self.check_surrounding_cells()
                # non_image_state = [percentage_explored + sublist for sublist in surroundings_state]
                non_image_state = surroundings_state.copy()
            location_map = np.zeros((self.nr,) + self.grid.shape)
            for r_i in range(self.nr):
                location_map[r_i][self.pos[r_i].y, self.pos[r_i].x] = 1

            position_array = np.zeros((self.nr,) + (self.grid.shape[0]*self.grid.shape[1],))
            for r_i in range(self.nr):
                position_array[r_i] = location_map[r_i].flatten()

            if self.encoding == "position":
                state = np.zeros(((self.nr,) + (self.grid.shape[0]*self.grid.shape[1],)))
                for r_i in range(self.nr):
                    state[r_i] = position_array.flatten()

                if self.lidar:
                    temp_state = state.copy()
                    state = np.zeros(((self.nr,) + (len(non_image_state[0])+self.grid.shape[0]*self.grid.shape[1],)))
                    for r_i in range(self.nr):
                        state[r_i] = np.concatenate((non_image_state[r_i], temp_state[r_i]), axis=0)

            # NB!!!!!!!!!!!!! other drone locations not included
            # should be included

            # Exploreation state: flattened concentrated arrays of environment where,
            # first array of environment position is equal to 1 and
            # second array of environment all explored cells are equal to 1
            if self.encoding == "position_exploration":
                exploration_grid = np.zeros(self.grid.shape)
                explored = np.argwhere(self.exploration_grid == True)
                for y,x in explored:
                    exploration_grid[y, x] = 1.0
                exploration_grid = exploration_grid.flatten()

                state = np.zeros(((self.nr,) + (self.grid.shape[0]*self.grid.shape[1]*2,)))

                for r_i in range(self.nr):
                    state[r_i] = np.concatenate((position_array[r_i], exploration_grid), axis=0)

                if self.lidar:
                    temp_state = state.copy()
                    state = np.zeros(((self.nr,) + (len(non_image_state[0])+self.grid.shape[0]*self.grid.shape[1]*2,)))
                    for r_i in range(self.nr):
                        state[r_i] = np.concatenate((non_image_state[r_i], temp_state[r_i]), axis=0)

            if self.encoding == "position_occupancy":
                occupancy_grid = np.zeros(self.grid.shape)
                explored = np.argwhere(self.exploration_grid == True)
                for y,x in explored:
                    occupancy_grid[y, x] = 0.2

                obstacles = np.argwhere(self.grid == States.OBS.value)
                for y,x in obstacles:
                    occupancy_grid[y, x] = 1.0

                occupancy_grid = occupancy_grid.flatten()

                state = np.zeros(((self.nr,) + (self.grid.shape[0]*self.grid.shape[1]*2,)))

                for r_i in range(self.nr):
                    state[r_i] = np.concatenate((position_array[r_i], occupancy_grid), axis=0)

                if self.lidar:
                    temp_state = state.copy()
                    state = np.zeros(((self.nr,) + (len(non_image_state[0])+self.grid.shape[0]*self.grid.shape[1]*2,)))
                    for r_i in range(self.nr):
                        state[r_i] = np.concatenate((non_image_state[r_i], temp_state[r_i]), axis=0)

            return state, non_image_state
                    
        # Image state
        elif "image" in self.encoding:
            location_map = np.zeros(((self.nr,) + self.grid.shape), dtype=np.float32)
            # if self.lidar:
            #     # percentage_explored = [np.mean(self.exploration_grid)]
            #     surroundings_state = self.check_surrounding_cells()
            #     # non_image_state = [percentage_explored + sublist for sublist in surroundings_state]
            #     non_image_state = surroundings_state.copy()

            if self.encoding == "image_occupancy":
                # Generates occupancy map. All explored cells are equal to 0.2 and obstacles as 1.0
                exploration_map = np.zeros(self.grid.shape)
                obstacle_map = np.zeros(self.grid.shape)
                explored = np.argwhere(np.logical_or(self.grid == States.EXP.value, self.grid == States.ROBOT.value))
                for y,x in explored:
                    exploration_map[y, x] = 1.0

                obstacles = np.argwhere(self.grid == States.OBS.value)
                for r_i in range(self.nr):
                    for y,x in obstacles:
                        obstacle_map[y, x] = 1.0

                # Cell test
                closest_unexplored = [None]*self.nr
                for r_i in range(self.nr):
                    if r_i == 0: other_location = None
                    else: other_location = closest_unexplored[r_i-1]
                    closest_unexplored[r_i] = self.get_closest_unexplored(r_i, other_location)
                closest_unexp_map = np.zeros((self.nr,) + self.grid.shape)
                if None not in closest_unexplored:
                    for r_i in range(self.nr):
                        closest_unexp_map[r_i][closest_unexplored[r_i].y, closest_unexplored[r_i].x] = 1.0
            
            elif self.encoding == "image":
                # Generates exploration map. All explored cells are equal to 1
                # shows exploration and obstacles as explored
                exploration_map = np.zeros(self.grid.shape)
                explored = np.argwhere(self.exploration_grid == True)
                for y,x in explored:
                    exploration_map[y, x] = 1

            elif self.encoding == "full_image":
                for r_i in range(self.nr):
                    explored = np.argwhere(self.exploration_grid == True)
                    for y,x in explored:
                        # location_map[r_i][y, x] = 0.5
                        location_map[r_i][y, x] = 0.1
                    obstacles = np.argwhere(self.grid == States.OBS.value)
                    for y,x in obstacles:
                        # location_map[r_i][y, x] = 0.01
                        location_map[r_i][y, x] = 0.5
                    if self.refuel: location_map[r_i][self.starting_pos[r_i].y, self.starting_pos[r_i].x] = 0.7


            # Generates location map. Position cell is equal to 1
            
            for r_i in range(self.nr):
                location_map[r_i][self.pos[r_i].y, self.pos[r_i].x] = 1.0

            # Generates other locations map. Other positions cells are equal to 1
            if self.nr > 1:
                drone_map = np.zeros((self.nr,) + (self.grid.shape))
                for r_i in range(self.nr):
                    for or_i in range(self.nr):
                        if or_i == r_i: continue
                        drone_map[r_i][self.pos[or_i].y, self.pos[or_i].x] = 1.0
                    

            # Generate image map
            # Padded
            # shape = list(self.grid.shape)
            # shape[0] += 2
            # shape[1] += 2

            # image_map = np.ones((self.nr,) + (3,) + tuple(shape))
            # for r_i in range(self.nr):
            #     image_map[r_i][0][1:shape[0]-1,1:shape[1]-1] = location_map[r_i]
            #     image_map[r_i][1][1:shape[0]-1,1:shape[1]-1] = other_locations_map[r_i]
            #     image_map[r_i][2][1:shape[0]-1,1:shape[1]-1] = exploration_map

            # Non-padded
            if not self.encoding == "full_image":
                image_map = np.zeros((self.nr,) + (2,) + self.grid.shape)
                if self.obstacles: image_map = np.zeros((self.nr,) + (3,) + self.grid.shape)
                if self.nr > 1:
                    # image_map = np.zeros((self.nr,) + (4,) + self.grid.shape)
                    image_map = np.zeros((self.nr,) + (3,) + self.grid.shape)
                    if self.obstacles: image_map = np.zeros((self.nr,) + (4,) + self.grid.shape)
                    # without exploration
                    # image_map = np.zeros((self.nr,) + (2,) + self.grid.shape)
                    if self.obstacles: image_map = np.zeros((self.nr,) + (3,) + self.grid.shape)
                for r_i in range(self.nr):
                    i = 0
                    image_map[r_i][i] = location_map[r_i]
                    if self.obstacles:
                        i += 1
                        image_map[r_i][i] = obstacle_map
                    # i += 1
                    # image_map[r_i][i] = obstacle_map
                    # i += 1
                    # image_map[r_i][i] = exploration_map
                    i += 1
                    image_map[r_i][i] = closest_unexp_map[r_i]
                    if self.nr > 1:
                        i += 1
                        image_map[r_i][i] = drone_map[r_i]
            elif self.encoding == "full_image":
                image_map = np.zeros((self.nr,) + (1,) + self.grid.shape)
                for r_i in range(self.nr):
                    image_map[r_i][0] = location_map[r_i]

            image_state = np.copy(image_map)
            # image_state = np.kron(image_state, np.ones((10, 10)))

            return image_state, non_image_state, closest_unexplored
        
        elif self.encoding == "local":
            closest_unexplored = [None]*self.nr

            closest_unexplored = self.scheduler()
            # for r_i in range(self.nr):
            #     # checks if enough fuel to get home
            #     if self.refuel and self.fuel-1 <= self.get_distance(self.pos[r_i], self.starting_pos[r_i]) and self.pos[r_i] != self.starting_pos[r_i]:
            #         closest_unexplored[r_i] = self.starting_pos[r_i]
            #         self.return_home = True
            #     else:
            #         if r_i == 0: other_location = None
            #         else: other_location = closest_unexplored[r_i-1]
            #         closest_unexplored[r_i] = self.cost_function(r_i, other_location)
            closest_unexp_map = np.zeros((self.nr,) + self.grid.shape)
            if None not in closest_unexplored:
                for r_i in range(self.nr):
                    closest_unexp_map[r_i][closest_unexplored[r_i].y, closest_unexplored[r_i].x] = 1.0
            
            state = [None]*self.nr
            for r_i in range(self.nr):
                if None not in closest_unexplored:
                    state[r_i] = [self.pos[r_i].x, self.pos[r_i].y, closest_unexplored[r_i].x, closest_unexplored[r_i].y] #+ self.check_surrounding_cells(r_i)
                else:
                    state[r_i] = [self.pos[r_i].x, self.pos[r_i].y, 0, 0] #+ self.check_surrounding_cells(r_i)
            # for r_i in range(self.nr):
            #     for i in range(self.nr):
            #         if i == r_i: continue
            #         state[r_i].append(self.pos[i].x)
            #         state[r_i].append(self.pos[i].y)
            
            
            return state, [None]*self.nr, closest_unexplored
        elif self.encoding == "local_drone":
            closest_unexplored = [None]*self.nr
            closest_unexplored_dir = [None]*self.nr

            closest_unexplored = self.scheduler()
            for ri in range(self.nr):
                closest_unexplored_dir[ri] = self.get_direction(closest_unexplored[ri], ri)
            
            state = [None]*self.nr
            for r_i in range(self.nr):
                state[r_i] = closest_unexplored_dir[r_i] + self.check_surrounding_cells(r_i)
            
            
            return state, [None]*self.nr, closest_unexplored

        # Goal state
        # goal_grid = np.zeros(self.grid.shape, dtype=np.float32)
        # goal_grid[self.goal.y, self.goal.x] = States.GOAL.value
        # goal_grid = goal_grid.flatten()

        # Obstacles state
        # obstacle_grid = np.zeros(self.grid.shape)
        # obstables = np.argwhere(self.grid == States.OBS.value)
        # for y,x in obstables:
        #     obstacle_grid[y, x] = 1.0
        # obstacle_grid = obstacle_grid.flatten()

    def is_obstacle_collision_centralized(self, pt, r_i, x ,y, action):
        # set new positions
        new_pos = [Point(x[i], y[i]) for i in range(self.nr)]

        # not needed. check later
        if r_i == None:
            next_r = 0
        else:
            next_r = r_i + 1
            if next_r % self.nr == 0: next_r = 0

        obstacles = np.argwhere(self.grid == States.OBS.value)
        # Collision with obstacle
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            self.score -= self.negative_reward
            self.collision['obstacle'][r_i] = True
            self.collision_state[r_i] = True
            x = self.pos[r_i].x
            y = self.pos[r_i].y
        return pt.x,pt.y

    def is_border_collision_centralized(self, pt, r_i, x ,y, action):
        # set new positions
        new_pos = [Point(x[i], y[i]) for i in range(self.nr)]

        # not needed. check later
        if r_i == None:
            next_r = 0
        else:
            next_r = r_i + 1
            if next_r % self.nr == 0: next_r = 0

        # Collision with boundary
        if not 0 <= pt.y < self.grid.shape[0] or not 0 <= pt.x < self.grid.shape[1]:
            self.score -= self.negative_reward
            self.collision['boundary'][r_i] = True
            self.collision_state[r_i] = True
            x = self.pos[r_i].x
            y = self.pos[r_i].y
        return pt.x,pt.y
    
    def is_drone_collision_centralized(self, pt, r_i, x ,y, action):
        # set new positions
        new_pos = [Point(x[i], y[i]) for i in range(self.nr)]

        # not needed. check later
        if r_i == None:
            next_r = 0
        else:
            next_r = r_i + 1
            if next_r % self.nr == 0: next_r = 0
        # Collision with other drone
        for i, pos_i in enumerate(new_pos):
            # collide at same location
            if any(self.collision_state):
                if (i != r_i and pt == self.pos[i]):
                    if action == Direction.STAY.value:
                        self.collision['trapped'][r_i] = True
                        self.collision_state[r_i] = True
                    else:
                        self.score -= self.negative_reward
                        self.collision['drone'][r_i] = True
                        self.collision['drone'][i] = True
                        self.collision_state[r_i] = True
            else:
                if (i != r_i and pt == pos_i):
                    if action == Direction.STAY.value:
                        self.collision['trapped'][r_i] = True
                        self.collision_state[r_i] = True
                    else:
                        self.score -= self.negative_reward
                        self.collision['drone'][r_i] = True
                        self.collision['drone'][i] = True
                        self.collision_state[r_i] = True

            # cross locations thus collide
            if i < r_i and pt == self.pos[i] and pos_i == self.pos[r_i]:
                self.score -= self.negative_reward
                self.collision['drone'][r_i] = True
                self.collision['drone'][i] = True
                self.collision_state[r_i] = True

    def is_collision_centralized(self, pt, r_i, x ,y, action):
        # set new positions
        new_pos = [Point(x[i], y[i]) for i in range(self.nr)]

        # not needed. check later
        if r_i == None:
            next_r = 0
        else:
            next_r = r_i + 1
            if next_r % self.nr == 0: next_r = 0

        obstacles = np.argwhere(self.grid == States.OBS.value)
        # Collision with obstacle
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            self.score -= self.negative_reward
            self.collision['obstacle'][r_i] = True
        # Collision with boundary
        if not 0 <= pt.y < self.grid.shape[0] or not 0 <= pt.x < self.grid.shape[1]:
            self.score -= self.negative_reward
            self.collision['boundary'][r_i] = True
        # Collision with other drone
        for i, pos_i in enumerate(new_pos):
            # collide at same location
            if i < r_i and pt == pos_i:
                if action == Direction.STAY.value:
                    self.collision['trapped'][r_i] = True
                else:
                    self.score -= self.negative_reward
                    self.collision['drone'][r_i] = True
                    self.collision['drone'][i] = True
            # cross locations thus collide
            if i < r_i and pt == self.pos[i] and pos_i == self.pos[r_i]:
                self.score -= self.negative_reward
                self.collision['drone'][r_i] = True
                self.collision['drone'][i] = True            
        
        return self.collision
    
    def _is_collision_decentralized(self, pt, r_i, x ,y):
        new_pos = [Point(x[i], y[i]) for i in range(self.nr)]
        if r_i == None:
            next_r = 0
        else:
            next_r = r_i + 1
            if next_r % self.nr == 0: next_r = 0

        obstacles = np.argwhere(self.grid == States.OBS.value)
        # Collision with obstacle
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            self.score[r_i] -= self.negative_reward
            self.collision['obstacle'][r_i] = True
            self.collision_state[r_i] = True
        # Collision with boundary
        elif not 0 <= pt.y < self.grid.shape[0] or not 0 <= pt.x < self.grid.shape[1]:
            self.score[r_i] -= self.negative_reward
            self.collision['boundary'][r_i] = True
            self.collision_state[r_i] = True
        # Collision with other drone
        for i, pos_i in enumerate(new_pos):
            if i != r_i and pt == pos_i:
                self.score[r_i] -= self.negative_reward
                self.collision['drone'][r_i] = True
                self.collision['drone'][i] = True
                self.collision_state[r_i] = True
        
        return self.collision
        
    def _update_env(self):
        # Update robot position(s) on grid
        for i in range(0, self.nr): self.grid[self.prev_pos[i].y,self.prev_pos[i].x] = States.EXP.value
        for i in range(0, self.nr): self.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
            

    def _move_centralized(self, action):
        # Get directions
        for i in range(0, self.nr):
            if action[i] == (Direction.LEFT).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.RIGHT).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.UP).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.DOWN).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.STAY).value:
                self.direction[i] = action[i]

        # Set temp x and y variables
        x = [None]*self.nr
        y = [None]*self.nr

        # Calculate new location
        for i in range(0, self.nr):
            x[i] = self.pos[i].x
            y[i] = self.pos[i].y
            if self.direction[i] == (Direction.RIGHT).value:
                x[i] += 1
            elif self.direction[i] == (Direction.LEFT).value:
                x[i] -= 1
            elif self.direction[i] == (Direction.DOWN).value:
                y[i] += 1
            elif self.direction[i] == (Direction.UP).value:
                y[i] -= 1

        # Set position to new location
        # for i in range(0, self.nr):
        #     x[i], y[i] = self.is_obstacle_collision_centralized(Point(x[i],y[i]), i, x, y, action[i])
        #     x[i], y[i] = self.is_border_collision_centralized(Point(x[i],y[i]), i, x, y, action[i])
        # for i in range(0, self.nr):
        #     self.is_drone_collision_centralized(Point(x[i],y[i]), i, x, y, action[i])
        for i in range(0, self.nr):
            self.collision = self.is_collision_centralized(Point(x[i],y[i]), i, x, y, action[i])
        
        collision_types = []
        for collision_type, collision_states in self.collision.items():
            if any(collision_states):
                collision_types.append(collision_type)
        
        for i in range(0, self.nr):
            for collisions in collision_types:
                if self.collision[collisions][i]:
                    if collisions == "obstacle" or collisions == "boundary":
                        self.prev_pos[i] = self.pos[i]
                        # self.pos[i] = Point(x[i],y[i])
                        self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
                        self.collision_state[i] = True
                    elif collisions == "drone":
                        self.prev_pos[i] = self.pos[i]
                        self.pos[i] = Point(x[i],y[i])
                        self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
                        self.collision_state[i] = True
                    elif collisions == "trapped":
                        self.prev_pos[i] = self.pos[i]
                        # self.pos[i] = Point(x[i],y[i])
                        self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
                        self.collision_state[i] = True
            if not self.collision_state[i]:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(x[i],y[i])
                self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True

    def _move_decentralized(self, action):
        # Get directions
        for i in range(0, self.nr):
            if action[i] == (Direction.LEFT).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.RIGHT).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.UP).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.DOWN).value:
                self.direction[i] = action[i]

        # Set temp x and y variables
        x = [None]*self.nr
        y = [None]*self.nr

        # Calculate new location
        for i in range(0, self.nr):
            x[i] = self.pos[i].x
            y[i] = self.pos[i].y
            if self.direction[i] == (Direction.RIGHT).value:
                x[i] += 1
            elif self.direction[i] == (Direction.LEFT).value:
                x[i] -= 1
            elif self.direction[i] == (Direction.DOWN).value:
                y[i] -= 1
            elif self.direction[i] == (Direction.UP).value:
                y[i] += 1

        # Set position to new location
        for i in range(0, self.nr):
            self.collision = self._is_collision_decentralized(Point(x[i],y[i]), i, x, y)
        
        collision_types = []
        for collision_type, collision_states in self.collision.items():
            if any(collision_states):
                collision_types.append(collision_type)
        
        for i in range(0, self.nr):
            for collisions in collision_types:
                if self.collision[collisions][i]:
                    if collisions == "obstacle" or collisions == "boundary":
                        self.pos[i] = self.pos[i]
                        self.prev_pos[i] = self.pos[i]
                        self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
                        self.collision_state[i] = True
                    elif collisions == "drone":
                        self.prev_pos[i] = self.pos[i]
                        self.pos[i] = Point(x[i],y[i])
                        self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
                        self.collision_state[i] = True
            if not self.collision_state[i]:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(x[i],y[i])
                self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True

    def _found_all_goals(self):
        goals_found = [False]*HEIGHT
        for row in range(self.clear_rows.shape[0]):
            if np.array([True for i in range(0, self.nr) if self.pos[i] == self.goal[row]]).any() == True:
                goals_found[row] = True
                self.score += 1
        if all(found for found in goals_found):
            return True
        else:
            return False
    
    def _found_a_goal(self):
        for i, goal_i in enumerate(self.goal):
            for r_i in range(self.nr):
                if self.pos[r_i] == goal_i:
                    del self.goal[i]
                    return True
        
    def draw_bounds(self, episode, percentage=0.0):
        # if episode < self.total_episodes//10:
        if not self.stage == 2:
            if episode < self.total_episodes//10 or self.stage == 0:
                self.grid = np.ones((HEIGHT, WIDTH))
                middle_rows = math.ceil(HEIGHT / 3)
                middle_columns = math.ceil(WIDTH / 3)
                self.grid[middle_rows:-middle_rows, middle_columns:-middle_columns] = 0
                if percentage > 98.0:
                    self.stage = 1
                    percentage = 0.0

            # elif self.total_episodes//10 <= episode < self.total_episodes//5 and percentage > 98.0:
            elif self.stage == 1:
                self.grid = np.ones((HEIGHT, WIDTH))
                middle_rows = HEIGHT // 4
                middle_columns = WIDTH // 4
                self.grid[middle_rows:-middle_rows, middle_columns:-middle_columns] = 0
                if percentage > 98.0:
                    self.stage = 2

        self.clear_rows = np.where(np.any(self.grid == States.UNEXP.value, axis=1))[0]
        if self.clear_rows.shape[0]:
            np.random.shuffle(self.clear_rows)
            self.clear_rows = self.clear_rows[:5]
        # breakpoint
        # if episode > self.total_episodes//5 then nothing

    def check_surrounding_cells(self, r_i):
        surroundings = []
        # Check if the surrounding cells are on the edge
        right_is_boundary = self.pos[r_i].x == WIDTH - 1
        left_is_boundary = self.pos[r_i].x == 0
        top_is_boundary = self.pos[r_i].y == 0
        bottom_is_boundary = self.pos[r_i].y == HEIGHT - 1

        surroundings.append(right_is_boundary or self.grid[self.pos[r_i].y][self.pos[r_i].x+1] == States.OBS.value or self.grid[self.pos[r_i].y][self.pos[r_i].x+1] == States.ROBOT.value)
        surroundings.append(left_is_boundary or self.grid[self.pos[r_i].y][self.pos[r_i].x-1] == States.OBS.value or self.grid[self.pos[r_i].y][self.pos[r_i].x-1] == States.ROBOT.value)
        surroundings.append(top_is_boundary or self.grid[self.pos[r_i].y-1][self.pos[r_i].x] == States.OBS.value or self.grid[self.pos[r_i].y-1][self.pos[r_i].x] == States.ROBOT.value)
        surroundings.append(bottom_is_boundary or self.grid[self.pos[r_i].y+1][self.pos[r_i].x] == States.OBS.value or self.grid[self.pos[r_i].y+1][self.pos[r_i].x] == States.ROBOT.value)

        return surroundings
    
    def draw_edges(self):
        # Set left and right edges to 2
        for row in self.grid:
            row[0] = States.OBS.value
            row[HEIGHT - 1] = States.OBS.value

        # Set top and bottom edges to 2
        self.grid[0] = [States.OBS.value] * HEIGHT
        self.grid[WIDTH - 1] = [States.OBS.value] * HEIGHT

    def remove_edges(self, grid):
        # Set left and right edges to 2
        for row in grid:
            row[0] = States.UNEXP.value
            row[HEIGHT - 1] = States.UNEXP.value

        # Set top and bottom edges to 2
        grid[0] = [States.UNEXP.value] * HEIGHT
        grid[WIDTH - 1] = [States.UNEXP.value] * HEIGHT

        return grid

    def random_maze(self, width, height, complexity=.75, density=.75):
        r"""Generate a random maze array. 
        
        It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
        is ``1`` and for free space is ``0``. 
        
        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        for i in range(density):
            x, y = np.random.randint(0, shape[1]//2 + 1) * 2, np.random.randint(0, shape[0]//2 + 1) * 2
            Z[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_,x_ = neighbours[np.random.randint(0, len(neighbours))]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
                        
        return Z.astype(int)

    # random grid generator (not that good, leaves long lines of obstacles in environment)
    def random_grid(self, rows=16, cols=16, obstacleProb=0.3):
        '''Return a 2D numpy array representing a grid of randomly placed
        obstacles (where the likelihood of any cell being an obstacle
        is given by obstacleProb) and randomized start/destination cells.
        '''
        obstacleGrid = np.random.random_sample((rows, cols))
        grid = States.UNEXP.value * np.zeros((rows, cols), dtype=np.int8)
        grid[obstacleGrid <= obstacleProb] = States.OBS.value

        return grid

    # random grid generator, based on game of life
    def do_simulation_step(self, old_map, death_limit=2, birth_limit=3):
        new_map = np.zeros((WIDTH, HEIGHT), dtype=bool)
        
        for x in range(WIDTH):
            for y in range(HEIGHT):
                nbs = self.count_alive_neighbours(old_map, x, y)
                
                if old_map[x, y]:
                    if nbs < death_limit:
                        new_map[x, y] = False
                    else:
                        new_map[x, y] = True
                else:
                    if nbs > birth_limit:
                        new_map[x, y] = True
                    else:
                        new_map[x, y] = False
        
        return new_map

    def initialise_map(self, map, chance_to_start_alive=0.2):        
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if random.random() < chance_to_start_alive:
                    map[x, y] = True
        
        return map

    def count_alive_neighbours(self, map, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbour_x = x + i
                neighbour_y = y + j
                
                if i == 0 and j == 0:
                    continue
                elif neighbour_x < 0 or neighbour_y < 0 or neighbour_x >= map.shape[0] or neighbour_y >= map.shape[1]:
                    count += 1
                elif map[neighbour_x, neighbour_y]:
                    count += 1
        
        return count

    def generate_map(self, number_of_steps=0):
        cell_map = np.zeros((WIDTH, HEIGHT), dtype=bool)
        cell_map = self.initialise_map(cell_map)
        
        for i in range(number_of_steps):
            cell_map = self.do_simulation_step(cell_map)

        grid = np.zeros((WIDTH, HEIGHT))

        for x in range(WIDTH):
            for y in range(HEIGHT):
                if cell_map[x,y]: grid[x,y] = States.OBS.value
                elif not cell_map[x,y]: grid[x,y] = States.UNEXP.value
        
        return grid
    
    # Find number of surrounding cells
    def surroundingCells(self, rand_wall, grid):
        s_cells = 0
        if (grid[rand_wall[0]-1][rand_wall[1]] == States.UNEXP.value):
            s_cells += 1
        if (grid[rand_wall[0]+1][rand_wall[1]] == States.UNEXP.value):
            s_cells += 1
        if (grid[rand_wall[0]][rand_wall[1]-1] == States.UNEXP.value):
            s_cells +=1
        if (grid[rand_wall[0]][rand_wall[1]+1] == States.UNEXP.value):
            s_cells += 1

        return s_cells

    def generate_maze(self, height, width):
        # Init variables
        grid = []

        # Denote all cells as unvisited
        for i in range(0, height):
            line = []
            for j in range(0, width):
                line.append(States.EXP.value)
            grid.append(line)

        # Randomize starting point and set it a cell
        starting_height = int(random.random()*height)
        starting_width = int(random.random()*width)
        if (starting_height == 0):
            starting_height += 1
        if (starting_height == height-1):
            starting_height -= 1
        if (starting_width == 0):
            starting_width += 1
        if (starting_width == width-1):
            starting_width -= 1

        # Mark it as cell and add surrounding walls to the list
        grid[starting_height][starting_width] = States.UNEXP.value
        walls = []
        walls.append([starting_height - 1, starting_width])
        walls.append([starting_height, starting_width - 1])
        walls.append([starting_height, starting_width + 1])
        walls.append([starting_height + 1, starting_width])

        # Denote walls in maze
        grid[starting_height-1][starting_width] = States.OBS.value
        grid[starting_height][starting_width - 1] = States.OBS.value
        grid[starting_height][starting_width + 1] = States.OBS.value
        grid[starting_height + 1][starting_width] = States.OBS.value

        while (walls):
            # Pick a random wall
            rand_wall = walls[int(random.random()*len(walls))-1]

            # Check if it is a left wall
            if (rand_wall[1] != 0):
                if (grid[rand_wall[0]][rand_wall[1]-1] == States.EXP.value and grid[rand_wall[0]][rand_wall[1]+1] == States.UNEXP.value):
                    # Find the number of surrounding cells
                    s_cells = self.surroundingCells(rand_wall, grid)

                    if (s_cells < 2):
                        # Denote the new path
                        grid[rand_wall[0]][rand_wall[1]] = States.UNEXP.value

                        # Mark the new walls
                        # Upper cell
                        if (rand_wall[0] != 0):
                            if (grid[rand_wall[0]-1][rand_wall[1]] != States.UNEXP.value):
                                grid[rand_wall[0]-1][rand_wall[1]] = States.OBS.value
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])


                        # Bottom cell
                        if (rand_wall[0] != height-1):
                            if (grid[rand_wall[0]+1][rand_wall[1]] != States.UNEXP.value):
                                grid[rand_wall[0]+1][rand_wall[1]] = States.OBS.value
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])

                        # Leftmost cell
                        if (rand_wall[1] != 0):	
                            if (grid[rand_wall[0]][rand_wall[1]-1] != States.UNEXP.value):
                                grid[rand_wall[0]][rand_wall[1]-1] = States.OBS.value
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])
                    

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)

                    continue

            # Check if it is an upper wall
            if (rand_wall[0] != 0):
                if (grid[rand_wall[0]-1][rand_wall[1]] == States.EXP.value and grid[rand_wall[0]+1][rand_wall[1]] == States.UNEXP.value):

                    s_cells = self.surroundingCells(rand_wall, grid)
                    if (s_cells < 2):
                        # Denote the new path
                        grid[rand_wall[0]][rand_wall[1]] = States.UNEXP.value

                        # Mark the new walls
                        # Upper cell
                        if (rand_wall[0] != 0):
                            if (grid[rand_wall[0]-1][rand_wall[1]] != States.UNEXP.value):
                                grid[rand_wall[0]-1][rand_wall[1]] = States.OBS.value
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])

                        # Leftmost cell
                        if (rand_wall[1] != 0):
                            if (grid[rand_wall[0]][rand_wall[1]-1] != States.UNEXP.value):
                                grid[rand_wall[0]][rand_wall[1]-1] = States.OBS.value
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])

                        # Rightmost cell
                        if (rand_wall[1] != width-1):
                            if (grid[rand_wall[0]][rand_wall[1]+1] != States.UNEXP.value):
                                grid[rand_wall[0]][rand_wall[1]+1] = States.OBS.value
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)

                    continue

            # Check the bottom wall
            if (rand_wall[0] != height-1):
                if (grid[rand_wall[0]+1][rand_wall[1]] == States.EXP.value and grid[rand_wall[0]-1][rand_wall[1]] == States.UNEXP.value):

                    s_cells = self.surroundingCells(rand_wall, grid)
                    if (s_cells < 2):
                        # Denote the new path
                        grid[rand_wall[0]][rand_wall[1]] = States.UNEXP.value

                        # Mark the new walls
                        if (rand_wall[0] != height-1):
                            if (grid[rand_wall[0]+1][rand_wall[1]] != States.UNEXP.value):
                                grid[rand_wall[0]+1][rand_wall[1]] = States.OBS.value
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])
                        if (rand_wall[1] != 0):
                            if (grid[rand_wall[0]][rand_wall[1]-1] != States.UNEXP.value):
                                grid[rand_wall[0]][rand_wall[1]-1] = States.OBS.value
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])
                        if (rand_wall[1] != width-1):
                            if (grid[rand_wall[0]][rand_wall[1]+1] != States.UNEXP.value):
                                grid[rand_wall[0]][rand_wall[1]+1] = States.OBS.value
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)


                    continue

            # Check the right wall
            if (rand_wall[1] != width-1):
                if (grid[rand_wall[0]][rand_wall[1]+1] == States.EXP.value and grid[rand_wall[0]][rand_wall[1]-1] == States.UNEXP.value):

                    s_cells = self.surroundingCells(rand_wall, grid)
                    if (s_cells < 2):
                        # Denote the new path
                        grid[rand_wall[0]][rand_wall[1]] = States.UNEXP.value

                        # Mark the new walls
                        if (rand_wall[1] != width-1):
                            if (grid[rand_wall[0]][rand_wall[1]+1] != States.UNEXP.value):
                                grid[rand_wall[0]][rand_wall[1]+1] = States.OBS.value
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])
                        if (rand_wall[0] != height-1):
                            if (grid[rand_wall[0]+1][rand_wall[1]] != States.UNEXP.value):
                                grid[rand_wall[0]+1][rand_wall[1]] = States.OBS.value
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])
                        if (rand_wall[0] != 0):	
                            if (grid[rand_wall[0]-1][rand_wall[1]] != States.UNEXP.value):
                                grid[rand_wall[0]-1][rand_wall[1]] = States.OBS.value
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)

                    continue

            # Delete the wall from the list anyway
            for wall in walls:
                if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                    walls.remove(wall)
            


        # Mark the remaining unvisited cells as walls
        for i in range(0, height):
            for j in range(0, width):
                if (grid[i][j] == States.EXP.value):
                    grid[i][j] = States.OBS.value

        # # Set entrance and exit
        # for i in range(0, width):
        #     if (grid[1][i] == States.UNEXP.value):
        #         grid[0][i] = States.ROBOT.value
        #         break

        # for i in range(width-1, 0, -1):
        #     if (grid[height-2][i] == States.UNEXP.value):
        #         grid[height-1][i] = States.EXIT.value
        #         break

        # Remove % of walls
        grid = self.remove_edges(grid)
        possible_indexes = np.argwhere(np.array(grid) == States.OBS.value)
        np.random.shuffle(possible_indexes)
        indices = possible_indexes[0:int(len(possible_indexes)*(1-self.obstacle_density))]
        for index in indices:
            grid[index[0]][index[1]] = States.UNEXP.value

        return np.array(grid)
    
    def set_obstacles(self):
        # Calculate the number of elements to be filled with 1's
        total_elements = int(HEIGHT/2) * int(WIDTH/2)
        num_ones_to_place = int(self.obstacle_density * total_elements)

        # Generate random indices to place 1's
        possible_indexes = np.argwhere(np.array(self.starting_grid) == States.UNEXP.value)
        np.random.shuffle(possible_indexes)
        indexes = possible_indexes[:num_ones_to_place]

        # Set the elements at the random indices to 1
        self.starting_grid[indexes[:, 0], indexes[:, 1]] = States.OBS.value
