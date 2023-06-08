from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math

# from sar_dqn_main import COL_REWARD

# Environment characteristics
HEIGHT = 4
WIDTH = 4

# DENSITY = 30 # percentage

# Direction states
class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

# Block states
class States(Enum):
    UNEXP = 0
    OBS = 4
    ROBOT = 1
    GOAL = 2
    EXP = 3

# Setup position variable of robot as point
Point = namedtuple('Point', 'x, y')


class Environment:
    
    def __init__(self, nr, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, training_type, encoding):
        self.nr = nr

        # Set robot(s) position
        self.pos = [Point(0,0)]*self.nr
        self.prev_pos = [Point(0,0)]*self.nr
        self.starting_pos = [Point(0,0)]*self.nr
        
        # Generates grid (Grid[y,x])
        self.grid = self.generate_grid()
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        self.goal_state = np.ones((HEIGHT, WIDTH), dtype=np.bool_)

        # Set robot(s) position
        self.pos = self.starting_pos

        self.positive_reward = positive_reward
        self.positive_exploration_reward = positive_exploration_reward
        self.negative_reward = negative_reward
        self.negative_step_reward = negative_step_reward

        self.unexplored = False
        self.collision = {  'obstacle' :   [False]*self.nr,
                            'boundary' :   [False]*self.nr,
                            'drone'    :   [False]*self.nr}
        self.collision_state = False

        self.training_type = training_type
        self.encoding = encoding

        # print("\nGrid size: ", self.grid.shape)

    def generate_grid(self):        
        # Ggenerate grid of zeros 
        grid = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        # Note: grid[y][x]
        # Generate obstacles
        # CODE GOES HERE

        # Static goal location spawning
        # self.goal = Point(0,0)
        # grid[self.goal.y, self.goal.x] = States.GOAL.value

        # Random goal location spawning
        indices = np.argwhere(grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.goal = Point(indices[0,1], indices[0,0])
        grid[self.goal.y, self.goal.x] = States.GOAL.value

        # Set robot(s) start position
        for i in range(0, self.nr):
            indices = np.argwhere(grid == States.UNEXP.value)
            np.random.shuffle(indices)
            self.starting_pos[i] = Point(indices[i,0], indices[i,1])
            grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()

        return grid

    def reset(self):
        self.collision = {  'obstacle' :   [False]*self.nr,
                            'boundary' :   [False]*self.nr,
                            'drone'    :   [False]*self.nr}
        self.collision_state = False
        # Clear all visited blocks
        self.grid.fill(0)
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)

        # Static goal location spawning
        # self.goal = Point(0,0)
        # grid[self.goal.y, self.goal.x] = States.GOAL.value

        # Random goal location spawning
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
                self.starting_pos[i] = Point(indices[i,1], indices[i,0])
                self.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()

        self.direction = [(Direction.RIGHT).value for i in range(self.nr)]
                
        self.score = 0
        
        state = self.get_state()

        return state

    def step_centralized(self, actions):
        self.score = 0

        # 2. Do action
        self._move_centralized(actions) # update the robot
            
        # 3. Update score and get state
        game_over = False           

        # 4. Update environment
        self._update_env()

        state = self.get_state()

        self.score += self.calc_reward_centralized()
        self.score -= self.negative_step_reward
        reward = self.score

        # 5. Check exit condition
        if self.collision_state:
            reward = self.score
            game_over = True
            return state, reward, game_over, 0

        if np.array([True for i in range(0, self.nr) if self.pos[i] == self.goal]).any() == True:
        # if (self.exploration_grid == self.goal_state).all():
            self.score += self.positive_reward
            reward = self.score
            game_over = True
            return state, reward, game_over, 1

        # 6. return game over and score
        return state, reward, game_over, 0
    
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

    def calc_reward_centralized(self):
        temp_arr = np.array([False]*self.nr)
        score = 0
        for i in range(0, self.nr):
            if self.exploration_grid[self.pos[i].y, self.pos[i].x] == False:
                temp_arr[i] = True
            if temp_arr[i] == True:
                score += self.positive_exploration_reward
        return score
    
    def calc_reward_decentralized(self):
        temp_arr = np.array([False]*self.nr)
        score = [0]*self.nr
        for i in range(0, self.nr):
            if self.exploration_grid[self.pos[i].y, self.pos[i].x] == False:
                temp_arr[i] = True
            if temp_arr[i] == True:
                score[i] += self.positive_exploration_reward
        return score

    def get_state(self):
        # Position state
        # position_grid = np.zeros(self.grid.shape)
        # position_grid[self.pos.y, self.pos.x] = 1.0
        # position_grid = position_grid.flatten()

        # # Exploreation state
        # exploration_grid = np.zeros(self.grid.shape)
        # explored = np.argwhere(self.exploration_grid == True)
        # for y,x in explored:
        #     exploration_grid[y, x] = 1.0
        # exploration_grid = exploration_grid.flatten()

        # Goal state
        # goal_grid = np.zeros(self.grid.shape, dtype=np.float32)
        # goal_grid[self.goal.y, self.goal.x] = States.GOAL.value
        # goal_grid = goal_grid.flatten()

        # Image state
        # Generates exploration map. All explored cells are 1
        exploration_map = np.zeros(self.grid.shape)
        explored = np.argwhere(self.exploration_grid == True)
        for y,x in explored:
            exploration_map[y, x] = 1

        # Generates location map. Position cell is equal to 1
        location_map = np.zeros((self.nr,) + self.grid.shape)
        for r_i in range(self.nr):
            location_map[r_i][self.pos[r_i].y, self.pos[r_i].x] = 1

        # Generates other locations map. Other positions cells are equal to 1
        other_locations_map = np.zeros((self.nr,) + self.grid.shape)
        for r_i in range(self.nr):
            for or_i in range(self.nr):
                if or_i == r_i: continue
                other_locations_map[r_i][self.pos[or_i].y, self.pos[or_i].x] = 1

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
        image_map = np.zeros((self.nr,) + (3,) + self.grid.shape)
        for r_i in range(self.nr):
            image_map[r_i][0] = location_map[r_i]
            image_map[r_i][1] = other_locations_map[r_i]
            image_map[r_i][2] = exploration_map

        # Inputs:
        # 1. Position state:
        # state = position_grid

        # 2. Position state + exploration state
        # state = np.concatenate((position_grid, exploration_grid), axis=0)  

        # 3. Image state:
        state = np.copy(image_map)

        return state

    def _is_collision_centralized(self, pt, r_i, x ,y):
        new_pos = [Point(x[i], y[i]) for i in range(self.nr)]
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
        elif not 0 <= pt.y < self.grid.shape[0] or not 0 <= pt.x < self.grid.shape[1]:
            self.score -= self.negative_reward
            self.collision['boundary'][r_i] = True
        # Collision with other drone
        for i, pos_i in enumerate(new_pos):
            if i != r_i and pt == pos_i:
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
        # Collision with boundary
        elif not 0 <= pt.y < self.grid.shape[0] or not 0 <= pt.x < self.grid.shape[1]:
            self.score[r_i] -= self.negative_reward
            self.collision['boundary'][r_i] = True
        # Collision with other drone
        for i, pos_i in enumerate(new_pos):
            if i != r_i and pt == pos_i:
                self.score[r_i] -= self.negative_reward
                self.collision['drone'][r_i] = True
                self.collision['drone'][i] = True
        
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
        for i in range(0, self.nr):
            self.collision = self._is_collision_centralized(Point(x[i],y[i]), i, x, y)
        
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
                        self.collision_state = True
                    elif collisions == "drone":
                        self.prev_pos[i] = self.pos[i]
                        self.pos[i] = Point(x[i],y[i])
                        self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
                        self.collision_state = True
            if not self.collision_state:
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
                y[i] += 1
            elif self.direction[i] == (Direction.UP).value:
                y[i] -= 1

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
                        self.collision_state = True
                    elif collisions == "drone":
                        self.prev_pos[i] = self.pos[i]
                        self.pos[i] = Point(x[i],y[i])
                        self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
                        self.collision_state = True
            if not self.collision_state:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(x[i],y[i])
                self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True