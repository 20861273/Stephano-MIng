import math
import numpy as np
import os
from collections import namedtuple


Point = namedtuple('Point', 'x, y')

b = [0,0,0,1,2,3,4]

c = [a for a in b if a == 0]

print(c)