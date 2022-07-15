from multiprocessing.connection import wait
from enum import Enum
from collections import namedtuple
import numpy as np
import keyboard
import time

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class States(Enum):
    UNEXP = 0
    OBS = 1
    ROBOT = 2
    TARGET = 3
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

class MazeGame:
    
    def __init__(self):
        self.grid = np.array([[1,1,1,1,1,1,1,1,1],
                            [1,States.ROBOT.value,1,0,0,0,0,0,3],
                            [1,0,0,0,1,0,1,0,1],
                            [1,1,1,0,1,0,1,0,1],
                            [1,0,1,0,0,0,1,0,1],
                            [1,0,1,0,1,0,1,0,1],
                            [1,0,1,0,1,0,1,0,1],
                            [1,0,1,0,1,0,1,0,1],
                            [1,0,1,0,1,0,1,0,1],
                            [1,0,1,0,1,0,0,0,1],
                            [1,1,1,1,1,1,1,1,1]])
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.pos = Point(1,1)
        
        self.score = 0
        self.exit = Point(8, 1)
        
    def play_step(self):
        print(self.grid,"\n")
        while True:
            # 1. collect user input
            if keyboard.is_pressed('w'):
                self.direction = Direction.UP
                break
            elif keyboard.is_pressed('s'):
                self.direction = Direction.DOWN
                break
            elif keyboard.is_pressed('d'):
                self.direction = Direction.RIGHT
                break
            elif keyboard.is_pressed('a'):
                self.direction = Direction.LEFT
                break
            elif keyboard.is_pressed('q'):
                game_over = True
                return game_over, self.score
        
        # 2. move
        self._move(self.direction) # update the robot
        time.sleep(0.5)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            
            return game_over, self.score
            
        # 4. reached exit or just move
        if self.pos == self.exit:
            self.score += 1
            game_over = True
            return game_over, self.score
        
        # 5. update ui and clock
        self._update_maze()
        # 6. return game over and score
        return game_over, self.score
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.pos
        # hits boundary
        obstacles = np.argwhere(self.grid == 1)
        if any(np.equal(obstacles,np.array([self.pos.y,self.pos.x])).all(1)):
            #print(self.pos)
            print("Boom!", self.pos, self.grid[self.pos.y,self.pos.x])
            return True
        elif self.pos.y < 0 or self.pos.y > self.grid.shape[0]-1 or self.pos.x < 0 or self.pos.x > self.grid.shape[1]-1:
            print("Boom out wall!")
            return True
            
        
        return False
        
    def _update_maze(self):
        # Update robot position(s) on grid
        self.grid[self.pos.y,self.pos.x] = States.ROBOT.value
        self.grid[self.prev_pos.y,self.prev_pos.x] = States.EXP.value

        # Update target position on grid
        self.grid[1,8] = States.TARGET.value
        
    def _move(self, direction):
        x = self.pos.x
        y = self.pos.y
        if direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.DOWN:
            y += 1
        elif direction == Direction.UP:
            y -= 1
            

        self.prev_pos = self.pos 
        self.pos = Point(x, y)
            

if __name__ == '__main__':
    print("Unexplored block = 0\n",
            "Obstacle = 1\n"
            "Robot = 2\n"
            "Exit = 3\n"
            "Explored block = 4")
    game = MazeGame()
    
    # # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)