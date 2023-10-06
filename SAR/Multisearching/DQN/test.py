from enclosed_space_checker import Enclosed_space_check
import numpy as np
from enum import Enum

class States(Enum):
    UNEXP = 0
    OBS = 1
    ROBOT = 4
    GOAL = 2
    EXP = 3

grid = np.array([[0,1,1,1,1,0],[0,1,1,0,1,0],[0,1,1,0,1,1],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0]])
print(grid)
print("\n")

ES = Enclosed_space_check(6, 6, grid, States)
grid = ES.enclosed_space_handler()
print(grid)