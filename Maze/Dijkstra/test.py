import numpy as np
import random
from enum import Enum
from collections import namedtuple
import string
import networkx as nx
import matplotlib.pyplot as plt

HEIGHT = 20
WIDTH = 20

Point = namedtuple('Point', 'x, y')

class States(Enum):
    UNEXP = 0
    OBS = 1
    ROBOT = 2
    EXIT = 3
    EXP = 4

class MazeAI():
    def __init__(self):

        self.grid  = np.zeros((HEIGHT,WIDTH), dtype=np.uint8)

        # Set start and goal positions
        s_quadrant = random.randint(0,3)
        g_quadrant = (s_quadrant+2) % 4
        quadrants = np.array([  [[self.grid.shape[1]/4*3-1,self.grid.shape[1]-1],   [0,self.grid.shape[0]/4-1]],
                                [[0,self.grid.shape[1]/4-1],                        [0,self.grid.shape[0]/4-1]],
                                [[0,self.grid.shape[1]/4-1],                        [self.grid.shape[0]/4*3-1,self.grid.shape[0]-1]],
                                [[self.grid.shape[1]/4*3-1,self.grid.shape[1]-1],   [self.grid.shape[0]/4*3-1,self.grid.shape[0]-1]]], dtype=int)

        self.starting_pos = Point(random.randint(quadrants[s_quadrant,0,0],quadrants[s_quadrant,0,1]),
                random.randint(quadrants[s_quadrant,1,0],quadrants[s_quadrant,1,1]))

        self.exit = Point(random.randint(quadrants[g_quadrant,0,0],quadrants[g_quadrant,0,1]),
                random.randint(quadrants[g_quadrant,1,0],quadrants[g_quadrant,1,1]))

        self.grid[self.starting_pos.y, self.starting_pos.x] = States.ROBOT.value
        self.grid[self.exit.y, self.exit.x] = States.EXIT.value

        obs_per_lim = 10
        
grid  = np.zeros((HEIGHT,WIDTH), dtype=np.uint8)
possible_indexes = np.argwhere(grid == 0)

print(len(possible_indexes))