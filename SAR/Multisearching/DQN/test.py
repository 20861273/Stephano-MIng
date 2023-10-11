from enclosed_space_checker import Enclosed_space_check
import numpy as np
from enum import Enum

t = [[1,1], [2], [3]]

if all(len(t[i]) == 1 for i in range(len(t))):
    print("hi")