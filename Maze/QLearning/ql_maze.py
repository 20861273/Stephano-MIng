from enum import Enum
from collections import namedtuple
from time import sleep
import numpy as np
import random

# Maze characteristics
HEIGHT = 10
WIDTH = 10

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

class MazeAI:
    
    def __init__(self):
        # init game state
        self.reset(0, 0)

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

        # Set entrance and exit
        for i in range(0, width):
            if (grid[1][i] == States.UNEXP.value):
                grid[0][i] = States.ROBOT.value
                break

        for i in range(width-1, 0, -1):
            if (grid[height-2][i] == States.UNEXP.value):
                grid[height-1][i] = States.EXIT.value
                break

        return np.array(grid)


    def reset(self, sim, episode):
        # print("Unexplored block = 0\n",
        #     "Obstacle = 1\n"
        #     "Robot = 2\n"
        #     "Exit = 3\n"
        #     "Explored block = 4")
        # init game state
        if sim == 0 and episode == 0:
            # Generates grid
            self.grid = self.generate_grid()

            # Sets all explored blocks to wall blocks
            # (generation can not convert all the walls since the exsisting walls blocks it off)
            exp_pos = np.argwhere(self.grid == States.EXP.value)
            for i in exp_pos:
                #print(i)
                #print(self.grid[i[0],i[1]])
                self.grid[i[0],i[1]] = States.OBS.value

            # Setup robot starting position
            self.prev_pos = Point(np.argwhere(self.grid == States.ROBOT.value)[0,1], np.argwhere(self.grid == States.ROBOT.value)[0,0])
            self.pos = self.prev_pos
            self.starting_pos = self.pos
            
            # Setup maze exit
            self.exit = Point(np.argwhere(self.grid == States.EXIT.value)[0,1], np.argwhere(self.grid == States.EXIT.value)[0,0])
        else:
            # Setup robot starting position
            self.grid[self.pos.y, self.pos.x] = States.UNEXP.value
            self.pos = self.starting_pos
            self.prev_pos = self.pos
            self.grid[self.pos.y, self.pos.x] = States.ROBOT.value

            # Setup maze exit
            self.grid[self.exit.y, self.exit.x] = States.EXIT.value
        
        #print(self.grid)
        self.direction = (Direction.RIGHT).value
                
        self.score = 0
        self.frame_iteration = 0

        state = self.get_state()

        return state
        
    def step(self, action):
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        # 2. move
        self._move(action) # update the robot
            
        # 3. check if game over
        reward = 0
        self.score -= 0.1
        game_over = False

        state = self.get_state()
        
        reward = self.score

        # 4. update maze
        self._update_maze()

        # 5. reached exit or just move
        if self.pos == self.exit:
            #self.score += 1
            reward = self.score
            game_over = True
            return state, reward, game_over, self.score
        
        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_state(self):
        return self.pos.x*self.grid.shape[1] + self.pos.y
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.pos
        # hits boundary
        obstacles = np.argwhere(self.grid == 1)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            return True
        elif pt.y < 0 or pt.y > self.grid.shape[0]-1 or pt.x < 0 or pt.x > self.grid.shape[1]-1:
            return True
        
        return False

    def _is_explored(self, pt=None):
        if pt is None:
            pt = self.pos
        # hits boundary
        explored = np.argwhere(self.grid == States.EXP.value)
        if any(np.equal(explored,np.array([self.pos.y,self.pos.x])).all(1)):
            #print(self.pos)
            return True
        
        return False
        
    def _update_maze(self):
        if self.frame_iteration == 0:
            # Update robot position(s) on grid
            self.grid[self.pos.y,self.pos.x] = States.ROBOT.value
        else:
            # Update robot position(s) on grid
            self.grid[self.prev_pos.y,self.prev_pos.x] = States.UNEXP.value
            self.grid[self.pos.y,self.pos.x] = States.ROBOT.value
            

    def _move(self, action):
        if action == (Direction.LEFT).value:
            self.direction = action
            #print(action, (Direction.LEFT).value, self.direction)
        elif action == (Direction.RIGHT).value:
            self.direction = action
            #print(action, (Direction.RIGHT).value, self.direction)
        elif action == (Direction.UP).value:
            self.direction = action
            #print(action, (Direction.UP).value, self.direction)
        elif action == (Direction.DOWN).value:
            self.direction = action
            #print(action, (Direction.DOWN).value, self.direction)

        x = self.pos.x
        y = self.pos.y
        if self.direction == (Direction.RIGHT).value:
            x += 1
            #print("RIGHT")
        elif self.direction == (Direction.LEFT).value:
            x -= 1
            #print("LEFT")
        elif self.direction == (Direction.DOWN).value:
            y += 1
            #print("DOWN")
        elif self.direction == (Direction.UP).value:
            y -= 1
            #print("UP")

        if self._is_collision(Point(x,y)):
            self.pos = self.pos
            self.prev_pos = self.pos
        else:
            self.prev_pos = self.pos
            self.pos = Point(x,y)
            #print(action, self.prev_pos, self.pos)

        #print(action, self.prev_pos, self.pos)