from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math

# Environment characteristics
HEIGHT = 4
WIDTH = 3
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
    OBS = 1
    ROBOT = 2
    GOAL = 3
    EXP = 4

# Setup position variable of robot as point
Point = namedtuple('Point', 'x, y')


class Environment:
    
    def __init__(self):
        # Generates grid (Grid[y,x])
        self.grid = self.generate_grid()

        # Set robot(s) position
        self.pos = self.starting_pos

        self.positive_reward = self.grid.shape[0]*self.grid.shape[1]

        print("\nGrid size: ", self.grid.shape)

    def generate_grid(self):        
        # Ggenerate grid of zeros 
        grid = np.zeros((HEIGHT, WIDTH))

        # Generate obstacles
        # CODE GOES HERE

        # Set robot(s) start position
        indices = np.argwhere(grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.starting_pos = Point(indices[0,1], indices[0,0])

        grid[self.starting_pos.y, self.starting_pos.x] = States.ROBOT.value

        # Set goal position at least 20% of grid size away from robot(s)
        distance_to = 0
        while distance_to < grid.shape[0]*0.2:
            indices = np.argwhere(grid == States.UNEXP.value)
            np.random.shuffle(indices)
            self.goal = Point(indices[0,1], indices[0,0])
            distance_to = math.sqrt((self.starting_pos.x - self.goal.x)**2 +
                            (self.starting_pos.y - self.goal.y)**2)
        
        grid[self.goal.y, self.goal.x] = States.GOAL.value

        return grid

    def reset(self):
        visited = np.argwhere(self.grid == States.EXP.value)
        for i in visited:
            self.grid[i[0], i[1]] = States.UNEXP.value

        # Setup agent
        # Clear all robot blocks
        robot = np.argwhere(self.grid == States.ROBOT.value)
        for i in robot:
            self.grid[i[0], i[1]] = States.UNEXP.value
        # Set new starting pos
        indices = np.argwhere(self.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.starting_pos = Point(indices[0,1], indices[0,0])
        self.pos = self.starting_pos
        self.prev_pos = self.pos
        self.grid[self.pos.y, self.pos.x] = States.ROBOT.value

        # Setup goal
        # Clear all goal blocks
        goal = np.argwhere(self.grid == States.GOAL.value)
        for i in goal:
            self.grid[i[0], i[1]] = States.UNEXP.value
        # Set new goal pos
        indices = np.argwhere(self.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.goal = Point(indices[0,1], indices[0,0])
        self.grid[self.goal.y, self.goal.x] = States.GOAL.value
    
        self.direction = (Direction.RIGHT).value
                
        self.score = 0
        self.frame_iteration = 0

        state = self.get_state()

        return state

    def step(self, action):
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        self.score = 0
        # 2. Do action
        self._move(action) # update the robot
            
        # 3. Update score and get state
        self.score -= 0.1
        game_over = False

        state = self.get_state()
        
        reward = self.score

        # 4. Update environment
        self._update_env()

        # 5. Check exit condition
        if self.pos == self.goal:
            self.score += self.positive_reward
            reward = self.score
            game_over = True
            return state, reward, game_over, self.score

        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_state(self):
        return np.array([self.pos.x*self.grid.shape[0] + self.pos.y])

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.pos
        # hits boundary
        obstacles = np.argwhere(self.grid == 1)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            return True
        elif pt.y < 0 or pt.y > self.grid.shape[0]-1 or pt.x < 0 or pt.x > self.grid.shape[1]-1:
            self.score -= 2
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
        else:
            self.prev_pos = self.pos
            self.pos = Point(x,y)