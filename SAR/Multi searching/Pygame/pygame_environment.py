import numpy as np
from enum import Enum

import pygame

# from pygame_solver import input_data

class Colours():
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

class Block_state(Enum):
    UNEXP = 0
    OBS = 1
    ROBOT = 2
    GOAL = 3
    EXP = 4


class Environment:
    def __init__(self, input_data):
        # environment
        self.height = input_data["environment"]["height"]
        self.width = input_data["environment"]["width"]
        self.grid = np.zeros((self.height, self.width))
        self.generate_obstacles()
        
        # pygame
        pygame.init()
        WINDOW_SIZE = [640, 640]
        self.margin = 5
        self.block_width = (WINDOW_SIZE[0]-(self.width*self.margin+self.margin))/self.width
        self.block_height = (WINDOW_SIZE[1]-(self.height*self.margin+self.margin))/self.height
        
        self.screen = pygame.display.set_mode(WINDOW_SIZE)

    def generate_obstacles(self):
        pass

    def reset_environment(self):
        self.screen.fill(Colours.BLACK)
    
        # Draw the grid
        for row in range(self.height):
            for column in range(self.width):
                color = Colours.WHITE
                if self.grid[row][column] == Block_state.ROBOT: color = Colours.GREEN
                if self.grid[row][column] == Block_state.GOAL: color = Colours.RED
                
                pygame.draw.rect(self.screen,
                                color,
                                [(self.margin + self.block_width) * column + self.margin,
                                (self.margin + self.block_height) * row + self.margin,
                                self.block_width,
                                self.block_height])

    def update_environment(self, nr, previous_position, current_position):
        for i in range(0, nr):
            self.grid[previous_position[i].y, previous_position[i].x] = Block_state.EXP.value
            color = Colours.BLUE
            pygame.draw.rect(self.screen,
                            color,
                            [(self.margin + self.block_width) * previous_position[i].y + self.margin,
                            (self.margin + self.block_height) * previous_position[i].x + self.margin,
                            self.block_width,
                            self.block_height])

            self.grid[current_position[i].y, current_position[i].x] = Block_state.ROBOT.value
            color = Colours.GREEN
            pygame.draw.rect(self.screen,
                            color,
                            [(self.margin + self.block_width) * current_position[i].y + self.margin,
                            (self.margin + self.block_height) * current_position[i].x + self.margin,
                            self.block_width,
                            self.block_height])
        pygame.display.flip()

# while not done:
#             for event in pygame.event.get():  # User did something
#                 if event.type == pygame.QUIT:  # If user clicked close
#                     done = True  # Flag that we are done so we exit this loop
#                 elif event.type == pygame.MOUSEBUTTONDOWN:
#                     # User clicks the mouse. Get the position
#                     pos = pygame.mouse.get_pos()
#                     # Change the x/y screen coordinates to grid coordinates
#                     column = int(pos[0] // (self.block_width + self.margin))
#                     row = int(pos[1] // (self.block_height + self.margin))
#                     # Set that location to one
#                     self.grid[row][column] = 1
#                     print("Click ", pos, "Grid coordinates: (x, y) = ", column, row)
#             clock = pygame.time.Clock()

#             self.screen.fill(Colours.BLACK)
    
#             # Draw the grid
#             for row in range(self.height):
#                 for column in range(self.width):
#                     color = Colours.WHITE
#                     if self.grid[row][column] == 1:
#                         color = Colours.GREEN
#                     pygame.draw.rect(self.screen,
#                                     color,
#                                     [(self.margin + self.block_width) * column + self.margin,
#                                     (self.margin + self.block_height) * row + self.margin,
#                                     self.block_width,
#                                     self.block_height])
        
#             # Limit to 60 frames per second
#             clock.tick(60)
        
#             # Go ahead and update the screen with what we've drawn.
#             pygame.display.flip()
    