from enclosed_space_checker import Enclosed_space_check
import numpy as np
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
import time
from collections import namedtuple
Point = namedtuple('Point', 'x, y')

d = np.array([[1,2,3],[4,5,6]])

print(np.minimum.reduceat(d[1,2,3,1,2,3,1,2,3,1,1,1,1,1,1,1,1,1],d[1,2,3,1,2,3,1,2,3,1,1,1,1,1,1,1,1,1],np.arange(0, 9, 3)))