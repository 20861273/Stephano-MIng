from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math
import time


# initialize list of lists
Point = namedtuple('Point', 'x, y')

states = np.zeros((4,3,4,3))
cnt = 0

for i in range(4):
    for j in range(3):
        for k in range(4):
            for l in range(3):
                states[i,j,k,l] = cnt
                cnt += 1

print(states)

for i in range(4):
    print("\n")
    for j in range(4):
            print(states[i,0,j], "  ", states[i,1,j], "  ", states[i,2,j])