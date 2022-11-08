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
    def __init__(self, nr):
        # Set robot(s) position
        self.pos = [0]*nr
        self.prev_pos = [0]*nr
        self.starting_pos = [0]*nr
        self.pos = self.starting_pos

        # Generates grid (Grid[y,x])
        self.grid = self.generate_grid(nr)

        print("\nGrid size: ", self.grid.shape)

    def generate_grid(self, nr):        
        # Ggenerate grid of zeros 
        grid = np.zeros((HEIGHT, WIDTH))

        # Generate obstacles
        # CODE GOES HERE

        # Distance code to keep for now
        # for i in range(0, nr):
        #     self.starting_pos[i] = Point(indices[i,1], indices[i,0])
        #     distance_to = 0
        #     next_i = i
        #     if not i == 0:
        #         while distance_to < grid.shape[0]*0.7:
        #             self.starting_pos[i] = Point(indices[next_i,1], indices[next_i,0])
        #             distance_to = math.sqrt((self.starting_pos[i].x - self.starting_pos[i-1].x)**2 +
        #                             (self.starting_pos[i].y - self.starting_pos[i-1].y)**2)
        #             next_i += 1

        #     grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        #     self.pos = self.starting_pos
        # Set robot(s) start position
        self.starting_pos = [0]*nr
        indices = np.argwhere(grid == States.UNEXP.value)
        np.random.shuffle(indices)
        i = 0
        self.starting_pos[0] = Point(indices[i,1], indices[i,0])
        self.starting_pos[1] = self.starting_pos[0]
        while not self.starting_pos[0].y < grid.shape[0]/2:
            self.starting_pos[0] = Point(indices[i,1], indices[i,0])
            i += 1
        while not self.starting_pos[1].y >= grid.shape[0]/2:
            self.starting_pos[1] = Point(indices[i,1], indices[i,0])
            i += 1

        for i in range(0, nr):
            grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
        self.pos = self.starting_pos

        # Set goal position
        indices = np.argwhere(grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.goal = Point(indices[0,1], indices[0,0])
        
        grid[self.goal.y, self.goal.x] = States.GOAL.value

        return grid