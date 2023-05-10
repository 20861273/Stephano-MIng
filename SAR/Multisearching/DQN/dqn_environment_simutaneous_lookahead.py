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
        self.goal_state = np.ones((HEIGHT, WIDTH), dtype=np.bool_)

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

    def reset(self):
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
        for i in range(0, self.nr):
            np.random.shuffle(indices)
            self.starting_pos[i] = Point(indices[i,1], indices[i,0])
            self.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        self.pos = self.starting_pos.copy()
        self.prev_pos = self.starting_pos.copy()

        self.direction = [(Direction.RIGHT).value for i in range(self.nr)]
                
        self.score = 0
        self.frame_iteration = 0
        
        state = self.get_state()

        return state

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

    def step(self, actions):
        # action = self.decode_action(action)
        self.frame_iteration += 1
        self.score = 0

        # 2. Do action
        self._move(actions) # update the robot
            
        # 3. Update score and get state
        game_over = False           

        # 4. Update environment
        self._update_env()

        state = self.get_state()

        self.score += self.calc_reward()
        reward = self.score

        # 5. Check exit condition
        # Goal state = missing person (uniform distribution)
        if np.array([True for i in range(0, self.nr) if self.pos[i] == self.goal]).any() == True:
        # Goal state = total coverage
        # if (self.exploration_grid == self.goal_state).all():
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

        # # Generates location map. Position cell is equal to 1
        # if selected_r == None:
        #     next_r = 0
        # else:
        #     next_r = selected_r + 1
        #     if next_r % self.nr == 0: next_r = 0
        # location_map = np.zeros(self.grid.shape)
        # location_map[self.pos[next_r].y, self.pos[next_r].x] = 1

        # # Generates other locations map. Other positions cells are equal to 1
        # other_locations_map = np.zeros(self.grid.shape)
        # for r_i in range(self.nr):
        #     if r_i == next_r: continue
        #     other_locations_map[self.pos[r_i].y, self.pos[r_i].x] = 1

        # Generate image map
        image_map = np.zeros((self.nr,) + (3,) + self.grid.shape)
        for r_i in range(self.nr):
            image_map[r_i][0] = location_map[r_i]
            image_map[r_i][1] = other_locations_map[r_i]
            image_map[r_i][2] = exploration_map
            breakpoint

        # Inputs:
        # 1. Position state:
        # state = position_grid

        # 2. Position state + exploration state
        # state = np.concatenate((position_grid, exploration_grid), axis=0)  

        # 3. Image state:
        state = np.copy(image_map)

        return state

    def get_next_state(self, action, exploration_grid, pos, next_r, current_r):
        # Make next move
        x, y = self._calc_location(action)

        # Set look ahead position to new location
        if self._is_collision(Point(x[current_r],y[current_r])):
            exploration_grid[pos[current_r].y, pos[current_r].x] = True
        else:
            exploration_grid[pos[current_r].y, pos[current_r].x] = True
            pos[current_r] = Point(x[current_r],y[current_r])

        # Image state
        # Generates exploration map. All explored cells are 1
        exploration_map = np.zeros(self.grid.shape)
        explored = np.argwhere(exploration_grid == True)
        for y,x in explored:
            exploration_map[y, x] = 1

        # Generates location map. Position cell is equal to 1
        location_map = np.zeros(self.grid.shape)
        location_map[pos[next_r].y, pos[next_r].x] = 1

        # Generates other locations map. Other positions cells are equal to 1
        other_locations_map = np.zeros(self.grid.shape)
        for r_i in range(self.nr):
            if r_i == next_r: continue
            other_locations_map[pos[r_i].y, pos[r_i].x] = 1

        # Generate image map
        image_map = np.zeros((3,) + self.grid.shape)
        image_map[0] = location_map
        image_map[1] = other_locations_map
        image_map[2] = exploration_map

        # Inputs:
        # 1. Position state:
        # state = position_grid

        # 2. Position state + exploration state
        # state = np.concatenate((position_grid, exploration_grid), axis=0)  

        # 3. Image state:
        state = np.copy(image_map)

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
        # if self.frame_iteration == 0:
        #     # Update robot position(s) on grid
        #     for i in range(0, self.nr):
        #         self.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
        # else:
        #     # Update robot position(s) on grid
        for i in range(0, self.nr): self.grid[self.prev_pos[i].y,self.prev_pos[i].x] = States.EXP.value
        for i in range(0, self.nr): self.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
            
    def _calc_location(self, action):
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
        return x, y
    
    def _move(self, action):
        x, y = self._calc_location(action)

        # Set position to new location
        for i in range(0, self.nr):
            if self._is_collision(Point(x[i],y[i])):
                self.pos[i] = self.pos[i]
                self.prev_pos[i] = self.pos[i]
                self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
            else:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(x[i],y[i])
                self.exploration_grid[self.prev_pos[i].y, self.prev_pos[i].x] = True
    