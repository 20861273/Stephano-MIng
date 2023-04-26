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
    
    def __init__(self, nr, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward):
        self.nr = nr

        # Set robot(s) position
        self.pos = [Point(0,0)]*self.nr
        self.prev_pos = [Point(0,0)]*self.nr
        self.starting_pos = [Point(0,0)]*self.nr
        
        # Generates grid (Grid[y,x])
        self.grid = self.generate_grid()
        self.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)

        # Set robot(s) position
        self.pos = self.starting_pos

        self.positive_reward = positive_reward
        self.positive_exploration_reward = positive_exploration_reward
        self.negative_reward = negative_reward
        self.negative_step_reward = negative_step_reward

        self.unexplored = False

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
        indices = np.argwhere(grid == States.UNEXP.value)
        np.random.shuffle(indices)
        for i in range(0, self.nr):
            self.starting_pos[i] = Point(indices[i,0], indices[i,1])
            grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()

        return grid

    def reset(self, last_start):
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
        indices = np.argwhere(self.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        if last_start != None:
            last_start += 1
            if last_start == HEIGHT*WIDTH-1:
                last_start = 0
            if last_start % WIDTH == self.goal.x and int(last_start/HEIGHT) == self.goal.y:
                last_start += 1
            if last_start == HEIGHT*WIDTH:
                last_start = 0
            x = last_start % WIDTH
            y = int(last_start/HEIGHT)
            self.starting_pos = Point(x, y)
        else:
            for i in range(0, self.nr):
                self.starting_pos[i] = Point(indices[i,0], indices[i,1])
                self.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()

        self.direction = [(Direction.RIGHT).value for i in range(self.nr)]
                
        self.score = 0
        self.frame_iteration = 0
        
        state = self.get_state()

        return state, last_start

        # visited = np.argwhere(self.grid == States.EXP.value)
        # for i in visited:
        #     self.grid[i[0], i[1]] = States.UNEXP.value

        # # Setup agent
        # # Clear all robot blocks
        # robot = np.argwhere(self.grid == States.ROBOT.value)
        # for i in robot:
        #     self.grid[i[0], i[1]] = States.UNEXP.value
        # # Set new starting pos
        # indices = np.argwhere(self.grid == States.UNEXP.value)
        # np.random.shuffle(indices)
        # self.starting_pos = Point(indices[0,1], indices[0,0])
        # self.pos = self.starting_pos
        # self.prev_pos = self.pos
        # self.grid[self.pos.y, self.pos.x] = States.ROBOT.value

        # # Setup goal
        # # Clear all goal blocks
        # goal = np.argwhere(self.grid == States.GOAL.value)
        # for i in goal:
        #     self.grid[i[0], i[1]] = States.UNEXP.value
        # # Set new goal pos
        # indices = np.argwhere(self.grid == States.UNEXP.value)
        # np.random.shuffle(indices)
        # self.goal = Point(indices[0,1], indices[0,0])
        # self.grid[self.goal.y, self.goal.x] = States.GOAL.value
    
        # self.direction = (Direction.RIGHT).value
                
        # self.score = 0
        # self.frame_iteration = 0

        # state = self.get_state()

        return state

    def step(self, action):
        action = self.decode_action(action)
        self.frame_iteration += 1
        self.score = 0

        # 2. Do action
        self._move(action) # update the robot
            
        # 3. Update score and get state
        game_over = False           

        # 4. Update environment
        self._update_env()

        state = self.get_state()

        self.score += self.calc_reward()
        reward = self.score

        # 5. Check exit condition
        if np.array([True for i in range(0, self.nr) if self.pos[i] == self.goal]).any() == True:
            self.score += self.positive_reward
            reward = self.score
            game_over = True
            return state, reward, game_over, self.score

        # 6. return game over and score
        return state, reward, game_over, self.score
    
    def decode_action(self, action):
        temp_action = action
        actions_completed = [0] * self.nr
        for i in range(self.nr):
            digit = temp_action % 4
            person = self.nr - i - 1
            actions_completed[person] = digit
            temp_action //= 4
        return actions_completed

    def calc_reward(self):
        temp_arr = np.array([False]*self.nr)
        score = 0
        for i in range(0, self.nr):
            if self.exploration_grid[self.pos[i].y, self.pos[i].x] == False:
                temp_arr[i] = True
            if temp_arr[i] == True:
                score += self.positive_exploration_reward
            else:
                score -= self.negative_step_reward
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
        image_grid = np.zeros(self.grid.shape)
        explored = np.argwhere(self.exploration_grid == True)
        # for y,x in explored:
        #     image_grid[y, x] = 0.5
        for i in range(0, self.nr): image_grid[self.pos[i].y, self.pos[i].x] = 1.0-float(i)/10
        

        # Inputs:
        # 1. Position state:
        # state = position_grid

        # 2. Position state + exploration state
        # state = np.concatenate((position_grid, exploration_grid), axis=0)  

        # 3. Image state:
        state = np.expand_dims(image_grid, axis=0)

        return state


    def _is_collision(self, pt):
        # hits boundary
        obstacles = np.argwhere(self.grid == States.OBS.value)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            return True
        elif not 0 <= pt.y < self.grid.shape[0] or not 0 <= pt.x < self.grid.shape[1]:
            self.score -= self.negative_reward
            return True
        
        return False

    def _is_explored(self, pt=None):
        if pt is None:
            pt = self.pos
        # hits boundary
        explored = np.argwhere(self.grid == States.EXP.value)
        if any(np.equal(explored,np.array([self.pos.y,self.pos.x])).all(1)):
            return True
        
        return False
        
    def _update_env(self):
        if self.frame_iteration == 0:
            # Update robot position(s) on grid
            for i in range(0, self.nr):
                self.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
        else:
            # Update robot position(s) on grid
            for i in range(0, self.nr): self.grid[self.prev_pos[i].y,self.prev_pos[i].x] = States.EXP.value
            for i in range(0, self.nr): self.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
            

    def _move(self, action):
        for i in range(0, self.nr):
            if action[i] == (Direction.LEFT).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.RIGHT).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.UP).value:
                self.direction[i] = action[i]
            elif action[i] == (Direction.DOWN).value:
                self.direction[i] = action[i]

        x = [None]*self.nr
        y = [None]*self.nr
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

        for i in range(0, self.nr):
            if self._is_collision(Point(x[i],y[i])):
                self.pos[i] = self.pos[i]
                self.prev_pos[i] = self.pos[i]
                self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
            else:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(x[i],y[i])
                self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True