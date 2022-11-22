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
        # distance_to = 0
        # while distance_to < grid.shape[0]*0.2:
        #     indices = np.argwhere(grid == States.UNEXP.value)
        #     np.random.shuffle(indices)
        #     self.goal = Point(indices[0,1], indices[0,0])
        #     distance_to = math.sqrt((self.starting_pos.x - self.goal.x)**2 +
        #                     (self.starting_pos.y - self.goal.y)**2)
        
        indices = np.argwhere(grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.goal = Point(indices[0,1], indices[0,0])
        grid[self.goal.y, self.goal.x] = States.GOAL.value

        return grid