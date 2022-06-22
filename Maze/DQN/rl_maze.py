from multiprocessing.connection import wait
from enum import Enum
from collections import namedtuple
from sre_constants import SUCCESS
import numpy as np
import time

class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class States(Enum):
    UNEXP = 0
    OBS = 1
    ROBOT = 2
    EXIT = 3
    EXP = 4


# Position state

    
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
        self.reset()

    def reset(self):
        # print("Unexplored block = 0\n",
        #     "Obstacle = 1\n"
        #     "Robot = 2\n"
        #     "Exit = 3\n"
        #     "Explored block = 4")
        # init game state
        self.grid = np.array([[1,1,1,1,1],
                            [1,States.ROBOT.value,1,0,States.EXIT.value],
                            [1,0,1,0,1],
                            [1,0,0,0,1],
                            [1,1,1,1,1]])
        
        
        self.direction = Direction.RIGHT
        
        self.prev_pos = Point(np.argwhere(self.grid == States.ROBOT.value)[0,1], np.argwhere(self.grid == States.ROBOT.value)[0,0])
        self.pos = self.prev_pos
        
        self.score = 0
        self.exit = Point(np.argwhere(self.grid == States.EXIT.value)[0,1], np.argwhere(self.grid == States.EXIT.value)[0,0])
        self.frame_iteration = 0
        return self.pos.x*self.grid.shape[1] + self.pos.y
        
    def step(self, action):
        self.frame_iteration += 1
        
        # 2. move
        self._move(action) # update the robot
        # 3. check if game over
        reward = 0
        self.score -= 1
        game_over = False

        if self.frame_iteration > 1000:
            reward = self.score
            game_over = True
            return reward, game_over, self.score
            
        # 4. reached exit or just move
        if self.pos == self.exit:
            print("Success")
            self.score += 1000
            reward = self.score
            game_over = True
            return reward, game_over, self.score
        
        # 5. update maze
        self._update_maze()

        # update reward

        reward = self.score
        
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            #print("hi")
            pt = self.pos
        # hits boundary
        obstacles = np.argwhere(self.grid == 1)
        #print(pt)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            #print(pt)
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
            #print(self.prev_pos.y,self.prev_pos.x)
        else:
            # Update robot position(s) on grid
            #print(self.prev_pos.y,self.prev_pos.x)
            self.grid[self.pos.y,self.pos.x] = States.ROBOT.value
            if self.grid[self.prev_pos.y,self.prev_pos.x] != States.OBS.value: self.grid[self.prev_pos.y,self.prev_pos.x] = States.EXP.value
        
    def _move(self, action):
        if action == (Direction.LEFT).value:
            self.direction = Direction.LEFT
            #print(action, (Direction.LEFT).value, self.direction)
        elif action == (Direction.RIGHT).value:
            self.direction = Direction.RIGHT
            #print(action, (Direction.RIGHT).value, self.direction)
        elif action == (Direction.UP).value:
            self.direction = Direction.UP
            #print(action, (Direction.UP).value, self.direction)
        elif action == (Direction.DOWN).value:
            self.direction = Direction.DOWN
            #print(action, (Direction.DOWN).value, self.direction)

        x = self.pos.x
        y = self.pos.y
        if self.direction == Direction.RIGHT:
            x += 1
            #print("RIGHT")
        elif self.direction == Direction.LEFT:
            x -= 1
            #print("LEFT")
        elif self.direction == Direction.DOWN:
            y += 1
            #print("DOWN")
        elif self.direction == Direction.UP:
            y -= 1
            #print("UP")
        
        if self.is_collision(Point(x,y)):
            self.prev_pos = self.prev_pos
            self.pos = self.prev_pos
        else:
            self.prev_pos = self.pos
            self.pos = Point(x,y)
            #print(action, self.prev_pos, self.pos)
        elif self.direction == Direction.UP:
            y -= 1
            #print("UP")
          
        self.prev_pos = self.pos
        self.pos = Point(x,y)
        #print(action, self.prev_pos, self.pos)
