from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math

# from sar_dqn_main import COL_REWARD

# Environment characteristics
HEIGHT = 8
WIDTH = 8

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
    
    def __init__(self, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward):
        # Generates grid (Grid[y,x])
        self.grid = self.generate_grid()

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
        self.starting_pos = Point(indices[0,1], indices[0,0])

        grid[self.starting_pos.y, self.starting_pos.x] = States.ROBOT.value

        return grid

    def reset(self, last_start):
        # Clear all visited blocks
        self.grid.fill(0)

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
            self.starting_pos = Point(indices[0,1], indices[0,0])
        self.pos = self.starting_pos
        self.prev_pos = self.pos
        self.grid[self.pos.y, self.pos.x] = States.ROBOT.value

        self.direction = (Direction.RIGHT).value
                
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
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        self.score = 0
        # 2. Do action
        self._move(action) # update the robot
            
        # 3. Update score and get state
        # self.score -= 0.1
        game_over = False

        # state = self.get_state()
        # goal_state = self.get_goal_state()
        # state = np.append(state, goal_state, axis=0)
           

        # 4. Update environment
        self._update_env()

        state = self.get_state()

        self.score += self.calc_reward()
        reward = self.score

        num_explored = len(np.argwhere(self.grid == States.EXP.value))

        # 5. Check exit condition
        if self.pos == self.goal:
        # if num_explored == WIDTH*HEIGHT:
            self.score += self.positive_reward
            reward = self.score
            game_over = True
            # cntr += 1
            # print(cntr)
            return state, reward, game_over, self.score#, cntr

        # 6. return game over and score
        return state, reward, game_over, self.score#, cntr

    def calc_reward(self):
        if self.unexplored:
            # grid = self.grid.copy()
            # grid[self.goal.y, self.goal.x] = States.UNEXP.value
            # grid[self.pos.y, self.pos.x] = States.UNEXP.value
            # explored = np.argwhere(self.grid == States.EXP.value)
            # return (len(explored)/(HEIGHT*WIDTH))*self.positive_reward
            return self.positive_exploration_reward
        else:
            return -self.negative_step_reward
    
    def get_position_state(self):
        grid = np.zeros(self.grid.shape)
        grid[self.pos.y, self.pos.x] = 1.0

        return grid.flatten()

    def get_goal_state(self):
        grid = np.zeros(self.grid.shape, dtype=np.float32)
        grid[self.goal.y, self.goal.x] = States.GOAL.value
        return grid.flatten()
    
    def get_state_unex(self):
        grid = np.zeros(self.grid.shape)
        explored = np.argwhere(self.grid == States.EXP.value)
        for y,x in explored:
            grid[y, x] = 1.0
        return grid.flatten()

    def get_state(self):
        # Position state
        position_grid = np.zeros(self.grid.shape)
        position_grid[self.pos.y, self.pos.x] = 1.0
        position_grid = position_grid.flatten()

        # Exploreation state
        exploration_grid = np.zeros(self.grid.shape)
        explored = np.argwhere(self.grid == States.EXP.value)
        for y,x in explored:
            exploration_grid[y, x] = 1.0
        exploration_grid = exploration_grid.flatten()

        # Goal state
        # goal_grid = np.zeros(self.grid.shape, dtype=np.float32)
        # goal_grid[self.goal.y, self.goal.x] = States.GOAL.value
        # goal_grid = goal_grid.flatten()

        # Image state
        image_grid = np.zeros(self.grid.shape)
        explored = np.argwhere(self.grid == States.EXP.value)
        for y,x in explored:
            image_grid[y, x] = 0.5
        image_grid[self.pos.y, self.pos.x] = 1.0
        

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
            self.grid[self.pos.y,self.pos.x] = States.ROBOT.value
        else:
            # Update robot position(s) on grid
            self.grid[self.prev_pos.y,self.prev_pos.x] = States.EXP.value
            self.grid[self.pos.y,self.pos.x] = States.ROBOT.value
            

    def _move(self, action):
        if action == (Direction.LEFT).value:
            self.direction = action
        elif action == (Direction.RIGHT).value:
            self.direction = action
        elif action == (Direction.UP).value:
            self.direction = action
        elif action == (Direction.DOWN).value:
            self.direction = action
        x = self.pos.x
        y = self.pos.y
        if self.direction == (Direction.RIGHT).value:
            x += 1
        elif self.direction == (Direction.LEFT).value:
            x -= 1
        elif self.direction == (Direction.DOWN).value:
            y += 1
        elif self.direction == (Direction.UP).value:
            y -= 1
        if self._is_collision(Point(x,y)):
            self.pos = self.pos
            self.prev_pos = self.pos
            self.unexplored = False
        else:
            self.unexplored = False
            self.prev_pos = self.pos
            self.pos = Point(x,y)
            if self.grid[self.pos.y, self.pos.x] == States.UNEXP.value:
                self.unexplored = True