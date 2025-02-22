from enum import Enum
from collections import namedtuple
import numpy as np
import random

# Maze characteristics
HEIGHT = 5
WIDTH = 6

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
    EXIT = 3
    EXP = 4

# Setup position variable of robot as point
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20
DENSITY = 30 # percentage

class Maze:
    
    def __init__(self):
        # Generates grid
        self.grid = self.generate_grid()
        self.pos = self.starting_pos
        self.grid[self.starting_pos.y, self.starting_pos.x] = States.ROBOT.value
        self.grid[self.exit.y, self.exit.x] = States.EXIT.value

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

    def generate_grid(self):
        # Init variables
        height = HEIGHT
        width = WIDTH
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
                if (grid[i][j] == 'u'):
                    grid[i][j] = States.OBS.value
        
        npgrid = np.array(grid)

        # Sets all explored blocks to wall blocks
        # (generation can not convert all the walls since the exsisting walls blocks it off)
        exp_pos = np.argwhere(npgrid == States.EXP.value)
        for i in exp_pos: npgrid[i[0],i[1]] = States.OBS.value
        
        # Set start and goal positions
        s_quadrant = random.randint(0,3)
        g_quadrant = (s_quadrant+2) % 4
        quadrants = np.array([  [[npgrid.shape[1]/4*3-1,npgrid.shape[1]-1],   [0,npgrid.shape[0]/4-1]],
                                [[0,npgrid.shape[1]/4-1],                        [0,npgrid.shape[0]/4-1]],
                                [[0,npgrid.shape[1]/4-1],                        [npgrid.shape[0]/4*3-1,npgrid.shape[0]-1]],
                                [[npgrid.shape[1]/4*3-1,npgrid.shape[1]-1],   [npgrid.shape[0]/4*3-1,npgrid.shape[0]-1]]], dtype=int)
        
        indices = np.argwhere(npgrid == States.OBS.value)
        self.starting_pos = Point(indices[0,1], indices[0,0])
        
        while npgrid[self.starting_pos.y, self.starting_pos.x] != States.UNEXP.value:
            self.starting_pos = Point(random.randint(quadrants[s_quadrant,0,0],quadrants[s_quadrant,0,1]),
                                    random.randint(quadrants[s_quadrant,1,0],quadrants[s_quadrant,1,1]))
        
        indices = np.argwhere(npgrid == States.OBS.value)
        self.exit = Point(indices[0,1], indices[0,0])
        while npgrid[self.exit.y, self.exit.x] != States.UNEXP.value:
            self.exit = Point(random.randint(quadrants[g_quadrant,0,0],quadrants[g_quadrant,0,1]),
                            random.randint(quadrants[g_quadrant,1,0],quadrants[g_quadrant,1,1]))

        npgrid[self.starting_pos.y, self.starting_pos.x] = States.ROBOT.value
        npgrid[self.exit.y, self.exit.x] = States.EXIT.value

        # Remove DENSITY % of walls
        possible_indexes = np.argwhere(npgrid == States.OBS.value)
        np.random.shuffle(possible_indexes)
        indices = possible_indexes[0:int(len(possible_indexes)*((100-DENSITY)/100))]
        for index in indices:
            npgrid[index[1], index[0]] = States.UNEXP.value

        return npgrid