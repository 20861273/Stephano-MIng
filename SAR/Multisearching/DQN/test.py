from enclosed_space_checker import Enclosed_space_check
import numpy as np
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
import time
from collections import namedtuple
Point = namedtuple('Point', 'x, y')

t = [True, True]

if not all(t):
    print(0)